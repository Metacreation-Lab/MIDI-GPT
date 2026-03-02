"""Generation smoke tests for the MIDI-GPT inference pipeline.

Requires a TorchScript checkpoint. Set MIDIGPT_CKPT env var to the .pt path.

Run on a GPU node:
    MIDIGPT_CKPT=/path/to/model.pt python3 -m pytest tests/test_inference.py -v -s
"""

import json
import os
import tempfile
from pathlib import Path

import pytest

try:
    import midigpt
except ImportError:
    pytest.skip("midigpt not installed", allow_module_level=True)

MIDI_TEST_DIR = Path(__file__).parent / "midi_files"
# Pick a small singletrack file for fast tests
MIDI_FILES = sorted(MIDI_TEST_DIR.glob("singletrack/*.mid"))
if not MIDI_FILES:
    MIDI_FILES = sorted(MIDI_TEST_DIR.glob("**/*.mid"))

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
    enc = _make_encoder()
    return json.loads(enc.midi_to_json(midi_path))


def _generate(piece_dict, ckpt, seed=-1, autoregressive=False):
    tracks = piece_dict["tracks"]
    num_bars = len(tracks[0].get("bars", []))

    status_tracks = []
    for ti, t in enumerate(tracks):
        tt = t.get("trackType", t.get("track_type", 0))
        n = len(t.get("bars", []))
        # Select middle bars for generation
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
            "autoregressive": autoregressive,
            "temperature": 0.9,
        })

    param = {
        "tracks_per_step": 1,
        "bars_per_step": min(2, num_bars),
        "model_dim": min(4, num_bars),
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
    result = midigpt.sample_multi_step(
        json.dumps(piece_dict),
        json.dumps({"tracks": status_tracks}),
        json.dumps(param),
        3,
        callbacks,
    )
    return json.loads(result[0])


@pytest.mark.inference
class TestGeneration:
    """Smoke tests for generation pipeline."""

    def test_generates_nonempty_output(self, ckpt_path, sample_piece_json):
        piece = json.loads(sample_piece_json)
        output = _generate(piece, ckpt_path, seed=42)
        # Output should have tracks with bars containing events
        assert "tracks" in output
        assert len(output["tracks"]) > 0
        total_events = len(output.get("events", []))
        assert total_events > 0, "Generated piece has no events"

    def test_deterministic_with_seed(self, ckpt_path, sample_piece_json):
        piece = json.loads(sample_piece_json)
        out1 = _generate(piece, ckpt_path, seed=42)
        out2 = _generate(piece, ckpt_path, seed=42)
        assert out1 == out2, "Same seed should produce identical output"

    def test_different_seeds_differ(self, ckpt_path, sample_piece_json):
        piece = json.loads(sample_piece_json)
        out1 = _generate(piece, ckpt_path, seed=42)
        out2 = _generate(piece, ckpt_path, seed=123)
        assert out1 != out2, "Different seeds should produce different output"

    def test_roundtrip_to_midi(self, ckpt_path, sample_piece_json):
        piece = json.loads(sample_piece_json)
        output = _generate(piece, ckpt_path, seed=42)
        enc = _make_encoder()
        with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as f:
            tmp_path = f.name
        try:
            enc.json_to_midi(json.dumps(output), tmp_path)
            assert os.path.getsize(tmp_path) > 0, "MIDI file is empty"
        finally:
            os.unlink(tmp_path)

    def test_autoregressive_mode(self, ckpt_path, sample_piece_json):
        piece = json.loads(sample_piece_json)
        output = _generate(piece, ckpt_path, seed=42, autoregressive=True)
        assert "tracks" in output
        assert len(output.get("events", [])) > 0
