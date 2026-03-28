"""Inference speed benchmarks for MIDI-GPT.

Measures tokens/sec and latency scaling across different generation sizes.

Run on a GPU node:
    MIDIGPT_CKPT=/path/to/model.pt python3 -m pytest tests/test_inference_benchmark.py -v -s
"""

import json
import time
from pathlib import Path

import pytest

try:
    import midigpt
except ImportError:
    pytest.skip("midigpt not installed", allow_module_level=True)

MIDI_TEST_DIR = Path(__file__).parent / "midi_files"

FROZEN_CONFIG = {
    "both_in_one": True,
    "unquantized": False,
    "do_multi_fill": False,
    "use_velocity_levels": True,
    "use_microtiming": True,
    "transpose": 0,
    "resolution": 12,
    "decode_resolution": 1920,
    "decode_final": False,
    "delta_resolution": 1920,
}


def _make_encoder():
    enc = midigpt.ExpressiveEncoder()
    for k, v in FROZEN_CONFIG.items():
        setattr(enc.config, k, v)
    return enc


def _load_piece(midi_path: str) -> dict:
    return json.loads(_make_encoder().midi_to_json(midi_path))


def _generate_timed(piece_dict, ckpt, seed=42, bars_per_step=2, model_dim=4):
    """Generate and return (output_dict, elapsed_seconds, n_output_tokens)."""
    tracks = piece_dict["tracks"]
    num_bars = len(tracks[0].get("bars", []))

    status_tracks = []
    for ti, t in enumerate(tracks):
        tt = t.get("trackType", t.get("track_type", 0))
        n = len(t.get("bars", []))
        selected = [False] * n
        if n > 2:
            for i in range(1, n - 1):
                selected[i] = True
        else:
            selected = [True] * n
        status_tracks.append({
            "track_id": ti,
            "track_type": tt if tt else 10,
            "ignore": False,
            "selected_bars": selected,
            "autoregressive": False,
            "temperature": 0.9,
        })

    param = {
        "tracks_per_step": 1,
        "bars_per_step": min(bars_per_step, num_bars),
        "model_dim": min(model_dim, num_bars),
        "percentage": 100,
        "batch_size": 1,
        "temperature": 0.9,
        "max_steps": 0,
        "polyphony_hard_limit": 0,
        "shuffle": False,
        "verbose": False,
        "ckpt": ckpt,
        "sampling_seed": seed,
        "mask_top_k": 0,
    }

    callbacks = midigpt.CallbackManager()

    start = time.perf_counter()
    result = midigpt.sample_multi_step(
        json.dumps(piece_dict),
        json.dumps({"tracks": status_tracks}),
        json.dumps(param),
        1,  # single attempt for clean timing
        callbacks,
    )
    elapsed = time.perf_counter() - start

    output = json.loads(result[0])

    # Count tokens by encoding the output
    enc = _make_encoder()
    try:
        tokens = enc.midi_to_tokens(json.dumps(output))
        n_tokens = len(tokens) if tokens else 0
    except Exception:
        n_tokens = -1

    return output, elapsed, n_tokens


# Collect MIDI files for benchmarking (skip very long ones)
_SKIP_STEMS = {
    "Maestro_1", "Maestro_2", "Maestro_3", "Maestro_4", "Maestro_5",
    "Maestro_6", "Maestro_7", "Maestro_8", "Maestro_9", "Maestro_10",
}
BENCH_FILES = sorted(
    f for f in MIDI_TEST_DIR.glob("**/*.mid")
    if f.stem not in _SKIP_STEMS and f.stem != "empty"
)


@pytest.mark.benchmark
@pytest.mark.inference
class TestTokensPerSecond:
    @pytest.mark.parametrize("midi_file", BENCH_FILES[:5],
                             ids=[f.stem for f in BENCH_FILES[:5]])
    def test_tokens_per_second(self, ckpt_path, midi_file):
        piece = _load_piece(str(midi_file))

        # Warm up
        _generate_timed(piece, ckpt_path, seed=0)

        # Timed runs
        n_runs = 3
        results = []
        for i in range(n_runs):
            _, elapsed, n_tokens = _generate_timed(piece, ckpt_path, seed=i + 1)
            tok_per_sec = n_tokens / elapsed if elapsed > 0 and n_tokens > 0 else 0
            results.append((elapsed, n_tokens, tok_per_sec))

        avg_latency = sum(r[0] for r in results) / n_runs
        avg_tokens = sum(r[1] for r in results) / n_runs
        avg_tps = sum(r[2] for r in results) / n_runs

        print(f"\n  [{midi_file.stem}] "
              f"avg_latency={avg_latency:.3f}s  "
              f"avg_tokens={avg_tokens:.0f}  "
              f"avg_tok/s={avg_tps:.1f}  "
              f"(n={n_runs})")


@pytest.mark.benchmark
@pytest.mark.inference
class TestLatencyScaling:
    def test_latency_vs_bars(self, ckpt_path):
        """Measure how latency scales with the number of generated bars."""
        # Use a multitrack file with enough bars
        multi_files = sorted(MIDI_TEST_DIR.glob("multitrack/*.mid"))
        if not multi_files:
            pytest.skip("No multitrack MIDI files found")

        # Find a file with enough bars
        piece = None
        for mf in multi_files:
            p = _load_piece(str(mf))
            n = len(p["tracks"][0].get("bars", []))
            if n >= 6:
                piece = p
                midi_name = mf.stem
                break
        if piece is None:
            pytest.skip("No MIDI file with >= 6 bars found")

        print(f"\n  Latency scaling for {midi_name}:")

        for bars_per_step in [1, 2, 4]:
            _, elapsed, n_tokens = _generate_timed(
                piece, ckpt_path, seed=42,
                bars_per_step=bars_per_step, model_dim=max(4, bars_per_step),
            )
            tok_per_sec = n_tokens / elapsed if elapsed > 0 and n_tokens > 0 else 0
            print(f"    bars_per_step={bars_per_step}: "
                  f"latency={elapsed:.3f}s  tokens={n_tokens}  tok/s={tok_per_sec:.1f}")
