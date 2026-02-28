"""Golden-output regression tests for the MIDI-GPT encoding pipeline.

These tests pin the encoding behavior at a fixed config so refactoring
changes that alter outputs are caught immediately.

Generate golden files (run once before refactoring):
    MIDIGPT_GENERATE_GOLDEN=1 python3 -m pytest tests/test_golden.py -v

Verify against golden files (run after each refactor change):
    python3 -m pytest tests/test_golden.py -v
"""

import json
import os
from pathlib import Path

import pytest

# Use the installed midigpt package directly (no CMake rebuild needed).
# This avoids the built_module fixture which requires CUDA/LibTorch on login nodes.
try:
    import midigpt
except ImportError:
    midigpt = None

GOLDEN_DIR = Path(__file__).parent / "golden"
MIDI_DIR = Path(__file__).parent.parent / "python_scripts_for_testing"
GENERATE = os.environ.get("MIDIGPT_GENERATE_GOLDEN", "0") == "1"

pytestmark = pytest.mark.skipif(midigpt is None, reason="midigpt not installed")

# Frozen config — matches ExpressiveEncoder() constructor defaults.
# Tests apply this explicitly so they test the algorithm, not config defaults.
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

# TrainConfig for the training pipeline test
FROZEN_TRAIN_CONFIG = {
    "num_bars": 4,
    "min_tracks": 1,
    "max_tracks": 8,
    "resolution": 12,
    "use_microtiming": True,
    "delta_resolution": 1920,
}


def _apply_encoder_config(enc, config_dict):
    """Explicitly set every config field to ensure reproducibility."""
    for key, value in config_dict.items():
        setattr(enc.config, key, value)


def _apply_train_config(tc, config_dict):
    """Explicitly set TrainConfig fields."""
    for key, value in config_dict.items():
        setattr(tc, key, value)


def _golden_path(name):
    return GOLDEN_DIR / f"{name}.json"


def _save(name, data):
    GOLDEN_DIR.mkdir(exist_ok=True)
    _golden_path(name).write_text(
        json.dumps(data, sort_keys=True, indent=2) + "\n"
    )


def _load(name):
    p = _golden_path(name)
    if not p.exists():
        pytest.fail(
            f"Golden file {p} not found. "
            "Run with MIDIGPT_GENERATE_GOLDEN=1 to create it."
        )
    return json.loads(p.read_text())


def _normalize_json_str(json_string):
    """Parse a protobuf JSON string to a Python dict for stable comparison."""
    return json.loads(json_string)


def _assert_or_save(name, actual):
    """In generate mode, save the golden file. In verify mode, compare."""
    if GENERATE:
        _save(name, actual)
        pytest.skip(f"Generated {name}.json")
    else:
        expected = _load(name)
        # Check config separately if present
        if "config" in actual and "config" in expected:
            assert actual["config"] == expected["config"], (
                f"Encoder config changed for {name}. "
                "If the default config was intentionally changed, "
                "update FROZEN_CONFIG and re-generate golden files."
            )
        # Check output
        output_key = [k for k in actual if k != "config"]
        for key in output_key:
            assert actual[key] == expected[key], (
                f"Output mismatch for '{key}' in {name}. "
                "Algorithm regression detected. "
                "If intentional, re-run with MIDIGPT_GENERATE_GOLDEN=1"
            )


def _make_encoder():
    """Create an ExpressiveEncoder with frozen config."""
    enc = midigpt.ExpressiveEncoder()
    _apply_encoder_config(enc, FROZEN_CONFIG)
    return enc


def _make_train_config():
    """Create a TrainConfig with frozen settings."""
    tc = midigpt.TrainConfig()
    _apply_train_config(tc, FROZEN_TRAIN_CONFIG)
    return tc


# ---------------------------------------------------------------------------
# Token & JSON golden tests
# ---------------------------------------------------------------------------


class TestGoldenTokens:
    def test_mtest_tokens(self):
        """Full forward pipeline: MIDI parse + encode for mtest.mid."""
        enc = _make_encoder()
        midi_path = str(MIDI_DIR / "mtest.mid")
        tokens = enc.midi_to_tokens(midi_path)
        _assert_or_save("mtest_tokens", {
            "config": FROZEN_CONFIG,
            "tokens": tokens,
        })

    def test_midigpt_gen_tokens(self):
        """Full forward pipeline: MIDI parse + encode for midigpt_gen.mid."""
        enc = _make_encoder()
        midi_path = str(MIDI_DIR / "midigpt_gen.mid")
        tokens = enc.midi_to_tokens(midi_path)
        _assert_or_save("midigpt_gen_tokens", {
            "config": FROZEN_CONFIG,
            "tokens": tokens,
        })

    def test_mtest_piece_json(self):
        """Protobuf serialization isolated from tokenization."""
        enc = _make_encoder()
        midi_path = str(MIDI_DIR / "mtest.mid")
        json_str = enc.midi_to_json(midi_path)
        _assert_or_save("mtest_piece", {
            "config": FROZEN_CONFIG,
            "piece": _normalize_json_str(json_str),
        })

    def test_mtest_roundtrip(self):
        """Roundtrip: json_to_tokens then tokens_to_json."""
        enc = _make_encoder()
        midi_path = str(MIDI_DIR / "mtest.mid")
        json_str = enc.midi_to_json(midi_path)
        tokens = enc.json_to_tokens(json_str)
        decoded_json_str = enc.tokens_to_json(tokens)
        _assert_or_save("mtest_roundtrip", {
            "config": FROZEN_CONFIG,
            "tokens": tokens,
            "decoded_piece": _normalize_json_str(decoded_json_str),
        })

    def test_training_pipeline_segment(self):
        """Dataset creation path: midi_to_json_bytes + select_random_segment."""
        tc = _make_train_config()
        midi_path = str(MIDI_DIR / "mtest.mid")
        raw = midigpt.midi_to_json_bytes(midi_path, tc, "{}")
        if not raw:
            pytest.skip("mtest.mid has no valid segments for this TrainConfig")
        json_str = midigpt.json_bytes_to_string(raw)
        segment = midigpt.select_random_segment(json_str, 4, 1, 4, 42)
        _assert_or_save("mtest_training_pipeline", {
            "config": FROZEN_CONFIG,
            "train_config": FROZEN_TRAIN_CONFIG,
            "segment": _normalize_json_str(segment),
        })


# ---------------------------------------------------------------------------
# Scalar golden tests
# ---------------------------------------------------------------------------


class TestGoldenScalars:
    def test_scalars(self):
        """Vocab size, max token, and pretty-print samples."""
        enc = _make_encoder()
        pretty_samples = {}
        for tok in [0, 1, 50, 100]:
            try:
                pretty_samples[str(tok)] = enc.pretty(tok)
            except Exception:
                pretty_samples[str(tok)] = None
        _assert_or_save("scalars", {
            "config": FROZEN_CONFIG,
            "vocab_size": enc.vocab_size(),
            "max_token": enc.rep.max_token(),
            "pretty_samples": pretty_samples,
        })
