"""Valid segments parity tests — verify segment selection is identical after optimizations.

These tests run midi_to_json_bytes() with various TrainConfig settings and
compare the internal_valid_segments output against golden references.

Workflow:
    1. Generate golden files from the pre-optimization build:
       MIDIGPT_GENERATE_GOLDEN=1 python3 -m pytest tests/test_valid_segments.py -v

    2. Apply optimizations, rebuild, then verify:
       python3 -m pytest tests/test_valid_segments.py -v
"""

import json
import os
from pathlib import Path

import pytest

try:
    import midigpt
except ImportError:
    midigpt = None

GOLDEN_DIR = Path(__file__).parent / "golden" / "valid_segments"
MIDI_MULTITRACK = sorted(
    (Path(__file__).parent / "midi_files" / "multitrack").glob("*.mid"))
MIDI_SINGLETRACK = sorted(
    (Path(__file__).parent / "midi_files" / "singletrack").glob("*.mid"))
MIDI_LEGACY = sorted(
    (Path(__file__).parent.parent / "python_scripts_for_testing").glob("*.mid"))
ALL_MIDI = MIDI_SINGLETRACK + MIDI_MULTITRACK + MIDI_LEGACY
GENERATE = os.environ.get("MIDIGPT_GENERATE_GOLDEN", "0") == "1"

pytestmark = pytest.mark.skipif(midigpt is None, reason="midigpt not installed")

# Test multiple TrainConfig settings to exercise different segment lengths
TRAIN_CONFIGS = [
    {"name": "4bar_1trk", "num_bars": 4, "min_tracks": 1, "max_tracks": 8,
     "resolution": 12, "use_microtiming": True, "delta_resolution": 1920},
    {"name": "8bar_2trk", "num_bars": 8, "min_tracks": 2, "max_tracks": 8,
     "resolution": 12, "use_microtiming": True, "delta_resolution": 1920},
    {"name": "2bar_1trk", "num_bars": 2, "min_tracks": 1, "max_tracks": 4,
     "resolution": 12, "use_microtiming": True, "delta_resolution": 1920},
]


def _make_train_config(config_dict):
    tc = midigpt.TrainConfig()
    for key, value in config_dict.items():
        if key == "name":
            continue
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


def _parse_with_config(midi_path, tc):
    """Parse a MIDI file and return the piece JSON including valid segments."""
    path_str = str(midi_path)
    try:
        raw = midigpt.midi_to_json_bytes(path_str, tc, "{}")
    except RuntimeError:
        return None
    if not raw:
        return None
    json_str = midigpt.json_bytes_to_string(raw)
    return json.loads(json_str)


def _assert_or_save(name, actual):
    if GENERATE:
        _save(name, actual)
        pytest.skip(f"Generated {name}.json")
    else:
        expected = _load(name)
        # Compare valid segments
        actual_segs = actual.get("piece", {}).get("internalValidSegments", [])
        expected_segs = expected.get("piece", {}).get("internalValidSegments", [])
        assert actual_segs == expected_segs, (
            f"Valid segments mismatch for {name}.\n"
            f"Expected {len(expected_segs)} segments, got {len(actual_segs)}."
        )
        # Compare valid tracks
        actual_tracks = actual.get("piece", {}).get("internalValidTracksV2", [])
        expected_tracks = expected.get("piece", {}).get("internalValidTracksV2", [])
        assert actual_tracks == expected_tracks, (
            f"Valid tracks mismatch for {name}."
        )


# ---------------------------------------------------------------------------
# Parametrized tests: all MIDI files × all TrainConfig variants
# ---------------------------------------------------------------------------

_test_cases = [
    (midi_file, tc_dict)
    for midi_file in ALL_MIDI
    for tc_dict in TRAIN_CONFIGS
]
_test_ids = [
    f"{_safe_stem(midi_file)}_{tc_dict['name']}"
    for midi_file, tc_dict in _test_cases
]


class TestValidSegments:
    @pytest.mark.parametrize("midi_file,tc_dict", _test_cases, ids=_test_ids)
    def test_segments(self, midi_file, tc_dict):
        tc = _make_train_config(tc_dict)
        result = _parse_with_config(midi_file, tc)
        if result is None:
            pytest.skip(f"{midi_file.name} has no valid segments for {tc_dict['name']}")
        _assert_or_save(
            f"segments_{_safe_stem(midi_file)}_{tc_dict['name']}",
            {"piece": result, "train_config": tc_dict},
        )
