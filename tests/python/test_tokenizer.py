"""Tests for `midigpt.tokenizer.tokenizer` (test plan section 3.4).

Covers:
  - Vocabulary invariants on a tokenizer built from `ghost_config`.
  - `Tokenizer.encode` output type / range / attribute-mutation semantics.
  - `encode → decode` roundtrip preserves pitches.
  - `resample_delta`: no-op fast-path, 12↔480 scaling, delta application,
    clamp-at-zero.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable

import pytest
from conftest import make_bar, make_note, melodic_track

import midigpt._core as _core
from midigpt._types import Bar, Note, Score, Track
from midigpt.tokenizer.tokenizer import Tokenizer, resample_delta


# --------------------------------------------------------------------------- #
#  vocab_size invariants
# --------------------------------------------------------------------------- #
def _nonzero_ranges(vocab: _core.Vocabulary) -> dict[str, tuple[int, int]]:
    out = {}
    for name, tt in _core.TokenType.__members__.items():
        if vocab.domain_size(tt) > 0:
            out[name] = vocab.range(tt)
    return out


def test_vocab_size_is_positive(ghost_tokenizer: Tokenizer) -> None:
    assert ghost_tokenizer.vocab_size() > 0


def test_vocab_size_covers_all_token_type_ranges(ghost_tokenizer: Tokenizer, ghost_config) -> None:
    """vocab_size must be exactly the max end of every populated domain range.

    The Vocabulary is laid out as contiguous, non-overlapping ranges starting
    at 0; therefore `vocab_size` == max(range.end) over all non-empty domains.
    """
    vocab = ghost_tokenizer._vocab
    ranges = _nonzero_ranges(vocab)
    assert ranges, "no populated token domains"
    starts = [s for s, _ in ranges.values()]
    ends = [e for _, e in ranges.values()]
    assert min(starts) == 0
    assert max(ends) == ghost_tokenizer.vocab_size()


def test_every_token_id_is_in_some_domain(ghost_tokenizer: Tokenizer) -> None:
    """The union of all non-empty domain ranges must cover [0, vocab_size)."""
    vocab = ghost_tokenizer._vocab
    covered = bytearray(ghost_tokenizer.vocab_size())
    for _, (s, e) in _nonzero_ranges(vocab).items():
        for i in range(s, e):
            covered[i] = 1
    assert all(covered), "some token ids lie outside any domain range"


# --------------------------------------------------------------------------- #
#  encode: type / value-range / attribute-mutation
# --------------------------------------------------------------------------- #
def test_encode_returns_list_of_ints(ghost_tokenizer: Tokenizer, simple_score: Score) -> None:
    tokens = ghost_tokenizer.encode(simple_score)
    assert isinstance(tokens, list)
    assert len(tokens) > 0
    assert all(isinstance(t, int) for t in tokens)


def test_encode_token_ids_within_vocab(ghost_tokenizer: Tokenizer, simple_score: Score) -> None:
    tokens = ghost_tokenizer.encode(simple_score)
    V = ghost_tokenizer.vocab_size()
    assert tokens, "expected at least one token"
    assert min(tokens) >= 0
    assert max(tokens) < V


def test_encode_compute_attributes_false_does_not_mutate_track(
    ghost_tokenizer: Tokenizer, simple_score: Score
) -> None:
    """With compute_attributes=False, track.attributes must be unchanged."""
    before = [dict(t.attributes) for t in simple_score.tracks]
    ghost_tokenizer.encode(simple_score, compute_attributes=False)
    after = [dict(t.attributes) for t in simple_score.tracks]
    assert before == after


def test_encode_compute_attributes_true_populates_track_attributes(
    ghost_tokenizer: Tokenizer, simple_score: Score
) -> None:
    """Sanity counter-check: compute_attributes=True DOES populate attrs.

    Without this, the previous test would be trivially satisfied by an encoder
    that never writes attributes at all.
    """
    before = dict(simple_score.tracks[0].attributes)
    ghost_tokenizer.encode(simple_score, compute_attributes=True)
    after = dict(simple_score.tracks[0].attributes)
    assert len(after) > len(before), (
        f"expected attributes to be populated; before={before} after={after}"
    )


# --------------------------------------------------------------------------- #
#  encode → decode roundtrip
# --------------------------------------------------------------------------- #
def _all_pitches(score: Score) -> list[int]:
    return [n.pitch for t in score.tracks for b in t.bars for n in b.notes]


def test_encode_decode_roundtrip_preserves_track_count(
    ghost_tokenizer: Tokenizer, simple_score: Score
) -> None:
    tokens = ghost_tokenizer.encode(simple_score)
    decoded = ghost_tokenizer.decode(tokens)
    assert isinstance(decoded, Score)
    assert len(decoded.tracks) == len(simple_score.tracks)


def test_encode_decode_roundtrip_preserves_pitches(
    ghost_tokenizer: Tokenizer, simple_score: Score
) -> None:
    original_pitches = sorted(_all_pitches(simple_score))
    assert original_pitches, "fixture has no notes"

    tokens = ghost_tokenizer.encode(simple_score)
    decoded = ghost_tokenizer.decode(tokens)
    decoded_pitches = sorted(_all_pitches(decoded))

    # Encoder may not generate notes in empty bars or compress identical
    # consecutive events, but every original pitch should appear (as a
    # multiset, original ⊆ decoded — usually equal).
    assert decoded_pitches == original_pitches, (
        f"pitch multiset changed: {original_pitches} -> {decoded_pitches}"
    )


# --------------------------------------------------------------------------- #
#  resample_delta
# --------------------------------------------------------------------------- #
def _make_zero_delta_score(res: int = 12) -> Score:
    """Single track, 1 bar, two notes, all deltas zero."""
    n1 = make_note(pitch=60, onset=0, dur=res, delta=0)
    n2 = make_note(pitch=64, onset=res, dur=res, delta=0)
    bar = make_bar([n1, n2])
    track = Track(bars=[bar], instrument=0, track_type="melodic")
    return Score(tracks=[track], resolution=res, tempo=500000)


def test_resample_delta_noop_returns_same_object_when_resolutions_match() -> None:
    """When source==target res AND all deltas are zero, returns score unchanged."""
    score = _make_zero_delta_score(res=12)
    onset_before = score.tracks[0].bars[0].notes[0].onset_ticks
    dur_before = score.tracks[0].bars[0].notes[0].duration_ticks

    out = resample_delta(score, source_res=12, target_res=12)

    # Fast-path: returns the same object (identity), nothing rewritten.
    assert out is score
    assert score.resolution == 12
    assert score.tracks[0].bars[0].notes[0].onset_ticks == onset_before
    assert score.tracks[0].bars[0].notes[0].duration_ticks == dur_before


def test_resample_delta_12_to_480_scales_onset_and_duration_by_40() -> None:
    """At resolution 12: onset=12, dur=12 -> at 480: onset=480, dur=480."""
    score = _make_zero_delta_score(res=12)
    out = resample_delta(score, source_res=12, target_res=480)

    assert out.resolution == 480
    notes = out.tracks[0].bars[0].notes
    assert notes[0].onset_ticks == 0
    assert notes[0].duration_ticks == 12 * 40
    assert notes[1].onset_ticks == 12 * 40
    assert notes[1].duration_ticks == 12 * 40
    # delta is reset to 0 after resampling.
    assert all(n.delta == 0 for n in notes)


def test_resample_delta_480_to_12_scales_down_and_floors() -> None:
    """480 -> 12 should divide by 40 (truncating). 481 -> 12."""
    score = _make_zero_delta_score(res=480)
    # Tweak a note to a non-multiple of 40 to exercise truncation.
    n = score.tracks[0].bars[0].notes[0]
    n.onset_ticks = 481  # → int(12 * 481 / 480) = int(12.025) = 12
    n.duration_ticks = 481  # → 12

    out = resample_delta(score, source_res=480, target_res=12)

    assert out.resolution == 12
    resampled = out.tracks[0].bars[0].notes[0]
    assert resampled.onset_ticks == 12
    assert resampled.duration_ticks == 12


def test_resample_delta_applies_delta_to_onset() -> None:
    """Positive delta shifts onset forward after the scale step."""
    score = _make_zero_delta_score(res=12)
    # Note 1 at onset=12 with delta=+5 -> at target 12 stays 12, +5 = 17.
    score.tracks[0].bars[0].notes[1].delta = 5

    out = resample_delta(score, source_res=12, target_res=12)

    # Note: noop fast-path only triggers when ALL deltas are zero; with a
    # nonzero delta we must take the rewrite path.
    n0, n1 = out.tracks[0].bars[0].notes
    assert n0.onset_ticks == 0
    assert n0.delta == 0
    assert n1.onset_ticks == 12 + 5
    assert n1.delta == 0  # delta consumed


def test_resample_delta_clamps_negative_at_zero() -> None:
    """A large negative delta must clamp resulting onset at 0, not go negative."""
    score = _make_zero_delta_score(res=12)
    # Onset 12 - 1000 would be -988; must clamp to 0.
    score.tracks[0].bars[0].notes[1].delta = -1000

    out = resample_delta(score, source_res=12, target_res=12)

    n1 = out.tracks[0].bars[0].notes[1]
    assert n1.onset_ticks == 0
    assert n1.delta == 0
