"""Tokenization parity tests — verify token output is identical after optimizations.

These tests run the full training pipeline path (parse → segment → tokenize)
on a broad set of MIDI files and compare token sequences against golden refs.

This covers what test_golden.py does for mtest.mid, but across all test files.

Workflow:
    1. Generate golden files from the pre-optimization build:
       MIDIGPT_GENERATE_GOLDEN=1 python3 -m pytest tests/test_tokenization_parity.py -v

    2. Apply optimizations, rebuild, then verify:
       python3 -m pytest tests/test_tokenization_parity.py -v
"""

import json
import os
from pathlib import Path

import pytest

try:
    import midigpt
except ImportError:
    midigpt = None

GOLDEN_DIR = Path(__file__).parent / "golden" / "tokenization_parity"
MIDI_MULTITRACK = sorted(
    (Path(__file__).parent / "midi_files" / "multitrack").glob("*.mid"))
MIDI_SINGLETRACK = sorted(
    (Path(__file__).parent / "midi_files" / "singletrack").glob("*.mid"))
MIDI_LEGACY = sorted(
    (Path(__file__).parent.parent / "python_scripts_for_testing").glob("*.mid"))
ALL_MIDI = MIDI_SINGLETRACK + MIDI_MULTITRACK + MIDI_LEGACY
GENERATE = os.environ.get("MIDIGPT_GENERATE_GOLDEN", "0") == "1"

pytestmark = pytest.mark.skipif(midigpt is None, reason="midigpt not installed")

# Same frozen config as test_golden.py
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

FROZEN_TRAIN_CONFIG = {
    "num_bars": 4,
    "min_tracks": 1,
    "max_tracks": 8,
    "resolution": 12,
    "use_microtiming": True,
    "delta_resolution": 1920,
}

SEED = 42


def _make_encoder():
    enc = midigpt.ExpressiveEncoder()
    for key, value in FROZEN_CONFIG.items():
        setattr(enc.config, key, value)
    return enc


def _make_train_config():
    tc = midigpt.TrainConfig()
    for key, value in FROZEN_TRAIN_CONFIG.items():
        setattr(tc, key, value)
    return tc


def _golden_path(name):
    return GOLDEN_DIR / f"{name}.json"


def _save(name, data):
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
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


def _safe_stem(path):
    return path.stem.replace(" ", "_")


def _segment_and_tokenize(midi_path):
    """Full training pipeline: parse → segment → tokenize.

    Returns (segment_json, tokens) or None if the file has no valid segments.
    """
    tc = _make_train_config()
    enc = _make_encoder()
    path_str = str(midi_path)

    # Parse and find valid segments
    try:
        raw = midigpt.midi_to_json_bytes(path_str, tc, "{}")
    except RuntimeError:
        return None
    if not raw:
        return None

    # Select a deterministic segment
    json_str = midigpt.json_bytes_to_string(raw)
    segment_json = midigpt.select_random_segment(
        json_str,
        FROZEN_TRAIN_CONFIG["num_bars"],
        FROZEN_TRAIN_CONFIG["min_tracks"],
        FROZEN_TRAIN_CONFIG["max_tracks"],
        SEED,
    )

    # Tokenize the segment
    tokens = enc.json_to_tokens(segment_json)

    return {
        "segment": json.loads(segment_json),
        "tokens": tokens,
    }


def _assert_or_save(name, actual):
    if GENERATE:
        _save(name, actual)
        pytest.skip(f"Generated {name}.json")
    else:
        expected = _load(name)
        assert actual["tokens"] == expected["tokens"], (
            f"Token sequence mismatch for {name}. Optimization regression."
        )
        assert actual["segment"] == expected["segment"], (
            f"Segment JSON mismatch for {name}. Optimization regression."
        )


# ---------------------------------------------------------------------------
# All MIDI files: parse → segment → tokenize
# ---------------------------------------------------------------------------

class TestSegmentTokenization:
    @pytest.mark.parametrize("midi_file", ALL_MIDI,
                             ids=[_safe_stem(f) for f in ALL_MIDI])
    def test_segment_tokens(self, midi_file):
        result = _segment_and_tokenize(midi_file)
        if result is None:
            pytest.skip(f"{midi_file.name} has no valid segments")
        _assert_or_save(f"seg_tok_{_safe_stem(midi_file)}", {
            "config": FROZEN_CONFIG,
            "train_config": FROZEN_TRAIN_CONFIG,
            **result,
        })
