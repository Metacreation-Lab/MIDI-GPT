"""Backend parity tests — verify midifile and symusic produce identical output.

These tests parse a broad set of MIDI files (singletrack + multitrack) and
compare the protobuf JSON output against golden references generated with
the midifile backend.

Workflow:
    1. Build with midifile (default) and generate golden files:
       MIDIGPT_GENERATE_GOLDEN=1 python3 -m pytest tests/test_backend_parity.py -v

    2. Build with symusic and verify:
       PYTHONPATH=build_symusic python3 -m pytest tests/test_backend_parity.py -v

Run:
    module load StdEnv/2023 python/3.11.5 abseil/20230125.3 protobuf/24.4
    python3 -m pytest tests/test_backend_parity.py -v
"""

import json
import os
from pathlib import Path

import pytest

try:
    import midigpt
except ImportError:
    midigpt = None

GOLDEN_DIR = Path(__file__).parent / "golden" / "backend_parity"
MIDI_MULTITRACK = sorted(
    (Path(__file__).parent / "midi_files" / "multitrack").glob("*.mid"))
MIDI_SINGLETRACK = sorted(
    (Path(__file__).parent / "midi_files" / "singletrack").glob("*.mid"))
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


def _apply_encoder_config(enc, config_dict):
    for key, value in config_dict.items():
        setattr(enc.config, key, value)


def _make_encoder():
    enc = midigpt.ExpressiveEncoder()
    _apply_encoder_config(enc, FROZEN_CONFIG)
    return enc


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
            "Build with midifile backend and run with "
            "MIDIGPT_GENERATE_GOLDEN=1 to create it."
        )
    return json.loads(p.read_text())


def _safe_stem(path):
    """Filesystem-safe stem: replace spaces with underscores."""
    return path.stem.replace(" ", "_")


def _parse_midi(midi_path):
    """Parse a MIDI file and return the piece JSON dict.

    Tests the MIDI parsing layer (midi_io::ParseSong) without tokenization,
    since tokenization is encoder-dependent and already covered by
    test_golden.py.  Returns None if the file has no notes (e.g. empty.mid).
    """
    enc = _make_encoder()
    path_str = str(midi_path)
    try:
        json_str = enc.midi_to_json(path_str)
    except RuntimeError:
        # e.g. "MIDI FILE HAS NO NOTES" for empty.mid
        return None
    return {
        "piece": json.loads(json_str),
    }


def _assert_or_save(name, actual):
    if GENERATE:
        _save(name, actual)
        pytest.skip(f"Generated {name}.json")
    else:
        expected = _load(name)
        assert actual["piece"] == expected["piece"], (
            f"Piece JSON mismatch for {name}. Backend parity broken."
        )


# ---------------------------------------------------------------------------
# Singletrack MIDI files
# ---------------------------------------------------------------------------

class TestSingletrack:
    @pytest.mark.parametrize("midi_file", MIDI_SINGLETRACK,
                             ids=[_safe_stem(f) for f in MIDI_SINGLETRACK])
    def test_parse(self, midi_file):
        result = _parse_midi(midi_file)
        if result is None:
            pytest.skip(f"{midi_file.name} has no notes")
        _assert_or_save(f"single_{_safe_stem(midi_file)}", result)


# ---------------------------------------------------------------------------
# Multitrack MIDI files
# ---------------------------------------------------------------------------

class TestMultitrack:
    @pytest.mark.parametrize("midi_file", MIDI_MULTITRACK,
                             ids=[_safe_stem(f) for f in MIDI_MULTITRACK])
    def test_parse(self, midi_file):
        result = _parse_midi(midi_file)
        if result is None:
            pytest.skip(f"{midi_file.name} has no notes")
        _assert_or_save(f"multi_{_safe_stem(midi_file)}", result)
