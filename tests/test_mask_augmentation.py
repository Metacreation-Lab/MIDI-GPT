"""Tests for mask-bar augmentation and GhostEncoder.

Covers:
- GhostEncoder vocabulary includes TOKEN_MASK_BAR
- TrainConfig mask fields are exposed to Python
- Random and structured-future mask modes produce masked tokens
- Masking is skipped when do_mask_augmentation=False
- Mask encoding is deterministic given a fixed seed
- ExpressiveEncoder vocabulary does NOT include TOKEN_MASK_BAR (backwards compat)
"""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent.resolve()
MULTITRACK_DIR = ROOT / "tests" / "midi_files" / "multitrack"


def _first_multitrack_midi():
    mids = sorted(MULTITRACK_DIR.glob("*.mid"))
    if not mids:
        pytest.skip("No multitrack MIDI files found in tests/midi_files/multitrack/")
    return str(mids[0])


# ---------------------------------------------------------------------------
# GhostEncoder vocabulary
# ---------------------------------------------------------------------------


class TestGhostEncoderVocab:
    def test_ghost_encoder_has_mask_bar_token(self, built_module):
        m = built_module
        enc = m.getEncoder(m.getEncoderType("GHOST_ENCODER"))
        assert enc.rep.has_token_type(m.TOKEN_TYPE.MASK_BAR), (
            "GhostEncoder rep must include TOKEN_MASK_BAR"
        )

    def test_expressive_encoder_lacks_mask_bar_token(self, built_module):
        """Backwards compat: existing encoders must NOT gain TOKEN_MASK_BAR."""
        m = built_module
        enc = m.getEncoder(m.getEncoderType("EXPRESSIVE_ENCODER"))
        assert not enc.rep.has_token_type(m.TOKEN_TYPE.MASK_BAR), (
            "ExpressiveEncoder must NOT include TOKEN_MASK_BAR (backwards compat)"
        )



# ---------------------------------------------------------------------------
# TrainConfig Python bindings
# ---------------------------------------------------------------------------


class TestTrainConfigMaskFields:
    def test_defaults(self, built_module):
        tc = built_module.TrainConfig()
        assert tc.do_mask_augmentation is False
        assert abs(tc.mask_apply_probability - 0.0) < 1e-6
        assert tc.mask_type == 0
        assert abs(tc.mask_bar_fraction - 0.0) < 1e-6
        assert tc.mask_max_lookahead == 4

    def test_set_fields(self, built_module):
        tc = built_module.TrainConfig()
        tc.do_mask_augmentation = True
        tc.mask_apply_probability = 0.5
        tc.mask_type = 2
        tc.mask_bar_fraction = 0.25
        tc.mask_max_lookahead = 6
        assert tc.do_mask_augmentation is True
        assert abs(tc.mask_apply_probability - 0.5) < 1e-6
        assert tc.mask_type == 2
        assert abs(tc.mask_bar_fraction - 0.25) < 1e-6
        assert tc.mask_max_lookahead == 6


# ---------------------------------------------------------------------------
# Mask augmentation encoding behaviour
# ---------------------------------------------------------------------------


def _encode_piece_with_mask(built_module, midi_path, mask_type, seed):
    """Encode a MIDI file with GhostEncoder and mask augmentation enabled."""
    m = built_module
    enc_type = m.getEncoderType("GHOST_ENCODER")
    enc = m.getEncoder(enc_type)

    enc.config.do_mask_augmentation = True
    enc.config.mask_apply_probability = 1.0  # always apply so tests are deterministic
    enc.config.mask_type = mask_type
    enc.config.mask_bar_fraction = 0.9       # high max fraction → likely to mask something
    enc.config.mask_max_lookahead = 4
    enc.config.mask_seed = seed

    piece_json = enc.midi_to_json(midi_path)
    piece = json.loads(piece_json)
    # Need >= 2 tracks and >= 2 bars for structured-future to do anything useful.
    if len(piece.get("tracks", [])) < 2 or len(piece["tracks"][0].get("bars", [])) < 2:
        pytest.skip("MIDI file too small for mask augmentation test (need 2+ tracks, 2+ bars)")

    # GhostEncoder only accepts NUM_BARS in {4, 8}; trim to 4 bars per track
    # (mirrors what the Jagged/training pipeline does before calling encode).
    for track in piece.get("tracks", []):
        track["bars"] = track.get("bars", [])[:4]
    piece_json = json.dumps(piece)

    tokens = enc.json_to_tokens(piece_json)
    return tokens


class TestMaskAugmentation:
    def test_mask_token_appears_with_random_mode(self, built_module):
        m = built_module
        midi = _first_multitrack_midi()
        tokens = _encode_piece_with_mask(m, midi, mask_type=0, seed=42)
        mask_token_id = m.getEncoder(m.getEncoderType("GHOST_ENCODER")).rep.encode(
            m.TOKEN_TYPE.MASK_BAR, 0
        )
        assert mask_token_id in tokens, (
            "Expected at least one MASK_BAR token with random mode (p=0.9, seed=42)"
        )

    def test_mask_token_appears_with_structured_future_mode(self, built_module):
        m = built_module
        midi = _first_multitrack_midi()
        tokens = _encode_piece_with_mask(m, midi, mask_type=1, seed=42)
        mask_token_id = m.getEncoder(m.getEncoderType("GHOST_ENCODER")).rep.encode(
            m.TOKEN_TYPE.MASK_BAR, 0
        )
        assert mask_token_id in tokens, (
            "Expected at least one MASK_BAR token with structured-future mode (seed=42)"
        )

    def test_no_mask_tokens_when_disabled(self, built_module):
        m = built_module
        midi = _first_multitrack_midi()
        enc_type = m.getEncoderType("GHOST_ENCODER")
        enc = m.getEncoder(enc_type)
        enc.config.do_mask_augmentation = False
        piece_json = enc.midi_to_json(midi)
        piece = json.loads(piece_json)
        for track in piece.get("tracks", []):
            track["bars"] = track.get("bars", [])[:4]
        tokens = enc.json_to_tokens(json.dumps(piece))
        mask_token_id = enc.rep.encode(m.TOKEN_TYPE.MASK_BAR, 0)
        assert mask_token_id not in tokens, (
            "MASK_BAR tokens must not appear when do_mask_augmentation=False"
        )

    def test_deterministic_with_fixed_seed(self, built_module):
        m = built_module
        midi = _first_multitrack_midi()
        t1 = _encode_piece_with_mask(m, midi, mask_type=0, seed=7)
        t2 = _encode_piece_with_mask(m, midi, mask_type=0, seed=7)
        assert t1 == t2, "Same seed must produce identical mask pattern"

    def test_different_seeds_can_differ(self, built_module):
        m = built_module
        midi = _first_multitrack_midi()
        t1 = _encode_piece_with_mask(m, midi, mask_type=0, seed=1)
        t2 = _encode_piece_with_mask(m, midi, mask_type=0, seed=999)
        # With p=0.9 and two different seeds, outcomes are very likely to differ.
        # This is a probabilistic check; it could theoretically fail but is extremely unlikely.
        assert t1 != t2, "Different seeds should (almost always) produce different masks"
