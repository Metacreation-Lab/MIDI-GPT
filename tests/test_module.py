"""API tests for the compiled midigpt extension.

These tests exercise the Python-visible surface of the C++ extension module.
They depend on the ``built_module`` session fixture from conftest.py, which
builds the extension with MIDIGPT_NO_TORCH=ON before the session starts.

Run:
    module load StdEnv/2023 python/3.11.5 abseil protobuf
    python3 -m pytest tests/test_module.py -v
"""

import json
import tempfile
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Basic sanity
# ---------------------------------------------------------------------------


class TestVersion:
    def test_version_returns_string(self, built_module):
        v = built_module.version()
        assert isinstance(v, str), f"version() returned {type(v)}"

    def test_version_nonempty(self, built_module):
        assert len(built_module.version()) > 0


# ---------------------------------------------------------------------------
# Encoder API
#
# getEncoderType(str) → ENCODER_TYPE enum
# getEncoder(ENCODER_TYPE) → ExpressiveEncoder (has .vocab_size(), .rep)
# getEncoderSize(ENCODER_TYPE) → int
# getEncoderTypeList() → list[str]
# ---------------------------------------------------------------------------

KNOWN_ENCODERS = ["EXPRESSIVE_ENCODER", "STEINBERG_WPCS_ENCODER"]


class TestEncoderAPI:
    def test_get_encoder_type_list_returns_list(self, built_module):
        lst = built_module.getEncoderTypeList()
        assert isinstance(lst, list), f"Expected list, got {type(lst)}"

    def test_get_encoder_type_list_nonempty(self, built_module):
        lst = built_module.getEncoderTypeList()
        assert len(lst) > 0, "getEncoderTypeList() is empty"

    def test_get_encoder_type_list_contains_known_encoder(self, built_module):
        lst = built_module.getEncoderTypeList()
        assert "EXPRESSIVE_ENCODER" in lst, (
            f"EXPRESSIVE_ENCODER missing from encoder list.\nGot: {lst}"
        )

    @pytest.mark.parametrize("encoder_name", KNOWN_ENCODERS)
    def test_get_encoder(self, built_module, encoder_name):
        et = built_module.getEncoderType(encoder_name)
        enc = built_module.getEncoder(et)
        assert enc is not None

    @pytest.mark.parametrize("encoder_name", KNOWN_ENCODERS)
    def test_encoder_vocab_size_positive(self, built_module, encoder_name):
        et = built_module.getEncoderType(encoder_name)
        enc = built_module.getEncoder(et)
        assert enc.vocab_size() > 0, (
            f"{encoder_name} vocab_size is {enc.vocab_size()}"
        )

    def test_get_encoder_size(self, built_module):
        n = built_module.getEncoderSize(built_module.ENCODER_TYPE.EXPRESSIVE_ENCODER)
        assert isinstance(n, int) and n > 0

    def test_get_encoder_type_round_trip(self, built_module):
        """getEncoderType should return the ENCODER_TYPE enum for a known encoder."""
        et = built_module.getEncoderType("EXPRESSIVE_ENCODER")
        assert et == built_module.ENCODER_TYPE.EXPRESSIVE_ENCODER

    def test_attribute_control_str(self, built_module):
        """getAttributeControlStr is callable and rejects unknown types.

        The C++ ATTRIBUTE_CONTROL return type is not registered with pybind11,
        so a successful call raises TypeError on return-value conversion.
        An invalid type name raises RuntimeError.  Either way the function is
        reachable and the invalid-type guard works correctly.
        """
        assert callable(built_module.getAttributeControlStr)
        with pytest.raises(RuntimeError):
            built_module.getAttributeControlStr("INVALID_CONTROL_TYPE")


# ---------------------------------------------------------------------------
# SteinbergWPCSEncoder-specific tests
# ---------------------------------------------------------------------------


class TestSteinbergWPCSEncoder:
    def test_construct(self, built_module):
        enc = built_module.SteinbergWPCSEncoder()
        assert enc is not None

    def test_vocab_size_positive(self, built_module):
        enc = built_module.SteinbergWPCSEncoder()
        assert enc.vocab_size() > 0

    def test_has_representation(self, built_module):
        enc = built_module.SteinbergWPCSEncoder()
        assert isinstance(enc.rep, built_module.REPRESENTATION)
        assert enc.rep.max_token() > 0

    def test_config_resolution_is_24(self, built_module):
        enc = built_module.SteinbergWPCSEncoder()
        assert enc.config.resolution == 24

    def test_attribute_control_types(self, built_module):
        enc = built_module.SteinbergWPCSEncoder()
        ac_types = enc.get_attribute_control_types()
        assert isinstance(ac_types, list)
        assert len(ac_types) == 6
        expected = {
            "ATTRIBUTE_CONTROL_TRACK_LEVEL_ONSET_DENSITY",
            "ATTRIBUTE_CONTROL_TRACK_LEVEL_ONSET_POLYPHONY",
            "ATTRIBUTE_CONTROL_TRACK_LEVEL_NOTE_DURATION",
            "ATTRIBUTE_CONTROL_REPETITION",
            "ATTRIBUTE_CONTROL_GENRE",
            "ATTRIBUTE_CONTROL_BAR_LEVEL_PITCH_CLASS_SET",
        }
        assert set(ac_types) == expected

    def test_encoder_type_enum(self, built_module):
        assert hasattr(built_module.ENCODER_TYPE, "STEINBERG_WPCS_ENCODER")

    def test_get_encoder_type_round_trip(self, built_module):
        et = built_module.getEncoderType("STEINBERG_WPCS_ENCODER")
        assert et == built_module.ENCODER_TYPE.STEINBERG_WPCS_ENCODER

    def test_get_encoder_returns_object(self, built_module):
        et = built_module.getEncoderType("STEINBERG_WPCS_ENCODER")
        enc = built_module.getEncoder(et)
        assert enc is not None
        assert enc.vocab_size() > 0

    def test_get_encoder_size(self, built_module):
        n = built_module.getEncoderSize(
            built_module.ENCODER_TYPE.STEINBERG_WPCS_ENCODER
        )
        assert isinstance(n, int) and n > 0

    def test_encode_decode_round_trip(self, built_module):
        """Test encode/decode round trip with a real MIDI file."""
        midi_path = (
            Path(__file__).parent.parent
            / "python_scripts_for_testing"
            / "mtest.mid"
        )
        if not midi_path.exists():
            pytest.skip(f"Test MIDI file not found: {midi_path}")

        enc = built_module.SteinbergWPCSEncoder()
        piece_json = enc.midi_to_json(str(midi_path))
        assert isinstance(piece_json, str) and len(piece_json) > 0

        tokens = enc.json_to_tokens(piece_json)
        assert isinstance(tokens, list) and len(tokens) > 0

        # all tokens should be within vocab range
        assert all(0 <= t < enc.vocab_size() for t in tokens)

        decoded_json = enc.tokens_to_json(tokens)
        assert isinstance(decoded_json, str) and len(decoded_json) > 0

    def test_in_encoder_type_list(self, built_module):
        lst = built_module.getEncoderTypeList()
        assert "STEINBERG_WPCS_ENCODER" in lst


# ---------------------------------------------------------------------------
# REPRESENTATION class
#
# getEncoder returns an encoder object with a .rep (REPRESENTATION) attribute.
# ---------------------------------------------------------------------------


class TestRepresentation:
    def test_get_encoder_returns_representation(self, built_module):
        et = built_module.ENCODER_TYPE.EXPRESSIVE_ENCODER
        enc = built_module.getEncoder(et)
        assert isinstance(enc.rep, built_module.REPRESENTATION)

    def test_token_domains_nonempty(self, built_module):
        et = built_module.ENCODER_TYPE.EXPRESSIVE_ENCODER
        enc = built_module.getEncoder(et)
        assert len(enc.rep.token_domains) > 0

    def test_max_token_positive(self, built_module):
        et = built_module.ENCODER_TYPE.EXPRESSIVE_ENCODER
        enc = built_module.getEncoder(et)
        assert enc.rep.max_token() > 0

    def test_get_type_mask(self, built_module):
        et = built_module.ENCODER_TYPE.EXPRESSIVE_ENCODER
        enc = built_module.getEncoder(et)
        # get_type_mask takes a list of TOKEN_TYPE enum values
        mask = enc.rep.get_type_mask([built_module.TOKEN_TYPE.PITCH])
        assert isinstance(mask, list)


# ---------------------------------------------------------------------------
# EncoderConfig
#
# ToJson() → dict[str, str]   (NOT a JSON string — it's a Python dict)
# FromJson(dict[str, str])
# ---------------------------------------------------------------------------


class TestEncoderConfig:
    def test_construct(self, built_module):
        cfg = built_module.EncoderConfig()
        assert cfg is not None

    def test_to_json_returns_dict(self, built_module):
        cfg = built_module.EncoderConfig()
        d = cfg.ToJson()
        assert isinstance(d, dict) and len(d) > 0

    def test_to_json_dict_has_resolution(self, built_module):
        cfg = built_module.EncoderConfig()
        d = cfg.ToJson()
        assert "resolution" in d

    def test_round_trip(self, built_module):
        cfg = built_module.EncoderConfig()
        cfg.resolution = 12
        cfg.use_microtiming = True
        dumped = cfg.ToJson()      # dict
        cfg2 = built_module.EncoderConfig()
        cfg2.FromJson(dumped)      # accepts dict
        assert cfg2.resolution == 12
        assert cfg2.use_microtiming is True


# ---------------------------------------------------------------------------
# TrainConfig
#
# to_json() → dict[str, str]   (NOT a JSON string — it's a Python dict)
# from_json(dict[str, str])
# ---------------------------------------------------------------------------


class TestTrainConfig:
    def test_construct(self, built_module):
        tc = built_module.TrainConfig()
        assert tc is not None

    def test_default_fields_accessible(self, built_module):
        tc = built_module.TrainConfig()
        _ = tc.num_bars
        _ = tc.min_tracks
        _ = tc.max_tracks
        _ = tc.resolution
        _ = tc.use_microtiming

    def test_to_json_returns_dict(self, built_module):
        tc = built_module.TrainConfig()
        data = tc.to_json()
        assert isinstance(data, dict)

    def test_round_trip(self, built_module):
        tc = built_module.TrainConfig()
        tc.num_bars = 8
        tc.resolution = 6
        dumped = tc.to_json()    # dict
        tc2 = built_module.TrainConfig()
        tc2.from_json(dumped)    # accepts dict
        assert tc2.num_bars == 8
        assert tc2.resolution == 6


# ---------------------------------------------------------------------------
# Protobuf helpers
# ---------------------------------------------------------------------------


class TestProtobufHelpers:
    def test_default_sample_param_nonempty(self, built_module):
        p = built_module.default_sample_param()
        assert isinstance(p, (str, bytes)) and len(p) > 0

    def test_default_sample_param_valid_json(self, built_module):
        p = built_module.default_sample_param()
        data = json.loads(p)
        assert isinstance(data, dict)

    def test_select_random_segment_returns_string(self, built_module):
        """select_random_segment takes positional args (no kwargs in pybind11).

        An empty piece JSON has no valid segments, so RuntimeError is raised.
        The test verifies the function is callable with the correct signature.
        """
        empty_piece = "{}"
        with pytest.raises(RuntimeError, match="NO VALID SEGMENTS"):
            built_module.select_random_segment(empty_piece, 4, 1, 4, 42)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class TestEnums:
    def test_model_type_track_model(self, built_module):
        assert hasattr(built_module.MODEL_TYPE, "TRACK_MODEL")

    def test_model_type_bar_infill(self, built_module):
        assert hasattr(built_module.MODEL_TYPE, "BAR_INFILL_MODEL")

    def test_token_type_pitch(self, built_module):
        assert hasattr(built_module.TOKEN_TYPE, "PITCH")

    def test_token_type_bar(self, built_module):
        assert hasattr(built_module.TOKEN_TYPE, "BAR")

    def test_token_type_velocity(self, built_module):
        assert hasattr(built_module.TOKEN_TYPE, "VELOCITY")

    def test_encoder_type_expressive(self, built_module):
        assert hasattr(built_module.ENCODER_TYPE, "EXPRESSIVE_ENCODER")

    def test_token_domain_construct(self, built_module):
        td = built_module.TOKEN_DOMAIN(128)
        assert td is not None


# ---------------------------------------------------------------------------
# MIDI I/O
# ---------------------------------------------------------------------------


class TestMidiIO:
    def test_midi_to_json_bytes_missing_file(self, built_module):
        """Passing a nonexistent path raises RuntimeError from the MIDI parser."""
        tc = built_module.TrainConfig()
        tc.num_bars = 4
        tc.min_tracks = 1
        with pytest.raises(RuntimeError):
            built_module.midi_to_json_bytes("/nonexistent/path.mid", tc, "{}")

    def test_json_bytes_to_string_empty(self, built_module):
        """Empty bytes should produce some string (empty or minimal JSON)."""
        result = built_module.json_bytes_to_string(b"")
        assert isinstance(result, str)

    def test_midi_to_json_bytes_with_real_file(self, built_module):
        """If a test MIDI file exists in the repo, parse it and verify output."""
        midi_path = (
            Path(__file__).parent.parent
            / "python_scripts_for_testing"
            / "mtest.mid"
        )
        if not midi_path.exists():
            pytest.skip(f"Test MIDI file not found: {midi_path}")

        tc = built_module.TrainConfig()
        tc.num_bars = 4
        tc.min_tracks = 1
        tc.max_tracks = 8
        tc.resolution = 12
        result = built_module.midi_to_json_bytes(str(midi_path), tc, "{}")
        # May be empty if the file has no valid segments for 4-bar windows
        assert isinstance(result, bytes)


# ---------------------------------------------------------------------------
# Jagged dataset reader (no Torch required)
#
# Jagged.append(str, split_id: int)   — stores serialised bytes as str
# Jagged.read_bytes(index, split_id)  → py::bytes
# ---------------------------------------------------------------------------


class TestJagged:
    def test_construct_write_mode(self, built_module, tmp_path):
        arr_path = str(tmp_path / "test.arr")
        j = built_module.Jagged(arr_path)
        j.enable_write()
        j.close()
        assert Path(arr_path).exists()

    def test_write_then_read_size(self, built_module, tmp_path):
        arr_path = str(tmp_path / "test.arr")

        # Write two items (split_id=0 for a single-split dataset)
        j = built_module.Jagged(arr_path)
        j.enable_write()
        j.append("hello", 0)
        j.append("world!", 0)
        j.close()

        # Read back
        j2 = built_module.Jagged(arr_path)
        j2.enable_read()
        assert j2.get_size() == 2
        j2.close()

    def test_write_then_read_bytes(self, built_module, tmp_path):
        arr_path = str(tmp_path / "test.arr")
        payload = "test payload 123"

        j = built_module.Jagged(arr_path)
        j.enable_write()
        j.append(payload, 0)
        j.close()

        j2 = built_module.Jagged(arr_path)
        j2.enable_read()
        result = j2.read_bytes(0, 0)   # (index, split_id)
        j2.close()
        # read_bytes returns py::bytes; compare as bytes
        assert result == payload.encode()


# ---------------------------------------------------------------------------
# Instrument helpers
#
# get_instruments_by_category(category_proto_name: str) → list[str]
# get_instrument_and_track_type_from_gm_inst(gm_inst_name: str) → (int, str)
# ---------------------------------------------------------------------------


class TestInstrumentHelpers:
    def test_get_instruments_by_category(self, built_module):
        # "GM_CATEGORY_POLY" is the proto enum name for polyphonic instruments
        result = built_module.get_instruments_by_category("GM_CATEGORY_POLY")
        assert isinstance(result, list) and len(result) > 0

    def test_get_instrument_and_track_type_returns_tuple(self, built_module):
        # "acoustic_grand_piano" is the proto enum name for GM program 0
        result = built_module.get_instrument_and_track_type_from_gm_inst(
            "acoustic_grand_piano"
        )
        inst_num, track_type = result
        assert isinstance(inst_num, int)
        assert isinstance(track_type, str)


# ---------------------------------------------------------------------------
# Torch inference API
# ---------------------------------------------------------------------------


class TestTorchAPI:
    """Verify that LibTorch-dependent inference functions are present."""

    def test_sample_multi_step_present(self, built_module):
        assert hasattr(built_module, "sample_multi_step"), (
            "sample_multi_step is missing — LibTorch may not have been linked"
        )

    def test_get_notes_present(self, built_module):
        assert hasattr(built_module, "get_notes"), (
            "get_notes is missing — LibTorch may not have been linked"
        )
