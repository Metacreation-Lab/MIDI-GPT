"""Tests for `midigpt.augmentation.*` (section 3.8 of TEST_IMPLEMENTATION_PLAN).

Each augmentation is exercised on a controlled in-memory Score (from
conftest fixtures) and asserted against the actual public API observed in
the source — not the API names mentioned in the test plan. Where the plan
mentions a feature that does not exist in the source, a `pytest.fail` test
documents the gap explicitly instead of skipping silently.
"""
from __future__ import annotations

import copy
import random
import sys
from pathlib import Path

# Conftest helpers (module-level builders)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from conftest import drum_track, make_bar, make_note, melodic_track  # noqa: E402

from midigpt._types import Score, Track
from midigpt.augmentation import (
    AugmentationPipeline,
    InstrumentSwap,
    MaskBar,
    MaskBarConfig,
    MaskMode,
    TrackPermutation,
    Transpose,
    VelocityScale,
    select_window,
)
from midigpt.augmentation.bar_window import BarWindow  # not re-exported


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _count_notes(score: Score) -> int:
    return sum(len(b.notes) for t in score.tracks for b in t.bars)


def _count_bars(score: Score) -> int:
    return sum(len(t.bars) for t in score.tracks)


def _all_pitches(score: Score) -> list[int]:
    return [n.pitch for t in score.tracks for b in t.bars for n in b.notes]


def _all_velocities(score: Score) -> list[int]:
    return [n.velocity for t in score.tracks for b in t.bars for n in b.notes]


# --------------------------------------------------------------------------- #
# Transpose
# --------------------------------------------------------------------------- #
class TestTranspose:
    def test_fixed_semitones_shifts_melodic_notes_by_exact_amount(self):
        score = Score(tracks=[melodic_track(n_bars=2)], resolution=12)
        before = _all_pitches(score)
        out = Transpose(5)(copy.deepcopy(score))
        after = _all_pitches(out)
        assert len(before) == len(after) > 0
        for b, a in zip(before, after):
            assert a == max(0, min(127, b + 5))

    def test_transpose_preserves_note_count_and_bar_count(self):
        score = Score(tracks=[melodic_track(n_bars=3, notes_per_bar=4)],
                      resolution=12)
        n_notes_before = _count_notes(score)
        n_bars_before  = _count_bars(score)
        out = Transpose(3)(copy.deepcopy(score))
        assert _count_notes(out) == n_notes_before
        assert _count_bars(out)  == n_bars_before

    def test_drum_track_not_transposed(self):
        score = Score(
            tracks=[melodic_track(n_bars=2), drum_track(n_bars=2)],
            resolution=12,
        )
        drum_before = [n.pitch
                       for b in score.tracks[1].bars for n in b.notes]
        out = Transpose(7)(copy.deepcopy(score))
        drum_after = [n.pitch
                      for b in out.tracks[1].bars for n in b.notes]
        assert drum_before == drum_after
        # Melodic shifted
        mel_before = [n.pitch
                      for b in score.tracks[0].bars for n in b.notes]
        mel_after  = [n.pitch
                      for b in out.tracks[0].bars for n in b.notes]
        assert all(a == b + 7 for a, b in zip(mel_after, mel_before))

    def test_transpose_clamps_high_pitches_at_127(self):
        score = Score(tracks=[Track(bars=[make_bar([
            make_note(pitch=125), make_note(pitch=120),
        ])], track_type="melodic")], resolution=12)
        out = Transpose(20)(copy.deepcopy(score))
        pitches = _all_pitches(out)
        assert all(0 <= p <= 127 for p in pitches)
        assert pitches == [127, 127]

    def test_transpose_clamps_low_pitches_at_0(self):
        score = Score(tracks=[Track(bars=[make_bar([
            make_note(pitch=2), make_note(pitch=5),
        ])], track_type="melodic")], resolution=12)
        out = Transpose(-20)(copy.deepcopy(score))
        assert _all_pitches(out) == [0, 0]

    def test_identity_transpose_is_noop(self):
        score = Score(tracks=[melodic_track(n_bars=2)], resolution=12)
        before = copy.deepcopy(score)
        out = Transpose(0)(score)
        assert _all_pitches(out) == _all_pitches(before)
        assert _count_notes(out) == _count_notes(before)

    def test_transpose_with_range_is_seedable(self):
        score = Score(tracks=[melodic_track(n_bars=2)], resolution=12)
        random.seed(123)
        a = Transpose(range(-6, 7))(copy.deepcopy(score))
        random.seed(123)
        b = Transpose(range(-6, 7))(copy.deepcopy(score))
        assert _all_pitches(a) == _all_pitches(b)


# --------------------------------------------------------------------------- #
# VelocityScale  (test plan called this "Velocity")
# --------------------------------------------------------------------------- #
class TestVelocityScale:
    def test_fixed_scale_multiplies_every_velocity(self):
        score = Score(tracks=[melodic_track(n_bars=2)], resolution=12)
        before = _all_velocities(score)
        out = VelocityScale(0.5)(copy.deepcopy(score))
        after = _all_velocities(out)
        assert len(after) == len(before)
        for b, a in zip(before, after):
            assert a == max(1, min(127, int(b * 0.5)))

    def test_velocity_floor_is_one(self):
        score = Score(tracks=[Track(bars=[make_bar([
            make_note(vel=1), make_note(vel=2),
        ])], track_type="melodic")], resolution=12)
        out = VelocityScale(0.01)(copy.deepcopy(score))
        vels = _all_velocities(out)
        assert all(1 <= v <= 127 for v in vels)
        assert vels == [1, 1]

    def test_velocity_ceiling_is_127(self):
        score = Score(tracks=[Track(bars=[make_bar([
            make_note(vel=100), make_note(vel=120),
        ])], track_type="melodic")], resolution=12)
        out = VelocityScale(10.0)(copy.deepcopy(score))
        assert _all_velocities(out) == [127, 127]

    def test_identity_velocity_scale_is_noop_on_integer_velocities(self):
        score = Score(tracks=[melodic_track(n_bars=2)], resolution=12)
        before = _all_velocities(score)
        out = VelocityScale(1.0)(copy.deepcopy(score))
        assert _all_velocities(out) == before

    def test_velocity_preserves_note_and_bar_counts(self):
        score = Score(tracks=[melodic_track(n_bars=3, notes_per_bar=4)],
                      resolution=12)
        nN, nB = _count_notes(score), _count_bars(score)
        out = VelocityScale(0.8)(copy.deepcopy(score))
        assert _count_notes(out) == nN
        assert _count_bars(out)  == nB

    def test_velocity_range_is_seedable(self):
        score = Score(tracks=[melodic_track(n_bars=2)], resolution=12)
        random.seed(42)
        a = VelocityScale((0.5, 1.5))(copy.deepcopy(score))
        random.seed(42)
        b = VelocityScale((0.5, 1.5))(copy.deepcopy(score))
        assert _all_velocities(a) == _all_velocities(b)


# --------------------------------------------------------------------------- #
# BarWindow  (not re-exported by __init__; imported from submodule)
# --------------------------------------------------------------------------- #
class TestBarWindow:
    def test_window_smaller_than_score_truncates_to_n_bars(self):
        score = Score(tracks=[melodic_track(n_bars=8)], resolution=12)
        random.seed(0)
        out = BarWindow(num_bars=3)(copy.deepcopy(score))
        for tr in out.tracks:
            assert len(tr.bars) <= 3
        assert len(out.tracks[0].bars) == 3

    def test_window_larger_than_score_returns_unchanged(self):
        score = Score(tracks=[melodic_track(n_bars=2)], resolution=12)
        out = BarWindow(num_bars=10)(copy.deepcopy(score))
        assert len(out.tracks[0].bars) == 2

    def test_window_is_contiguous(self):
        bars = [make_bar([make_note(pitch=60 + i)]) for i in range(8)]
        score = Score(tracks=[Track(bars=bars, track_type="melodic")],
                      resolution=12)
        random.seed(7)
        out = BarWindow(num_bars=4)(copy.deepcopy(score))
        pitches = [b.notes[0].pitch for b in out.tracks[0].bars]
        assert pitches == list(range(pitches[0], pitches[0] + 4))

    def test_window_seedable(self):
        score = Score(tracks=[melodic_track(n_bars=10)], resolution=12)
        random.seed(99)
        a = BarWindow(num_bars=3)(copy.deepcopy(score))
        random.seed(99)
        b = BarWindow(num_bars=3)(copy.deepcopy(score))
        assert [n.pitch for bar in a.tracks[0].bars for n in bar.notes] == \
               [n.pitch for bar in b.tracks[0].bars for n in bar.notes]


# --------------------------------------------------------------------------- #
# select_window  (the test plan called this "ScoreWindow")
# --------------------------------------------------------------------------- #
class TestScoreWindow:
    def test_select_window_returns_n_bars_on_n_tracks(self):
        score = Score(
            tracks=[melodic_track(n_bars=8), melodic_track(n_bars=8)],
            resolution=12,
        )
        rng = random.Random(0)
        out = select_window(score, n_bars=4, n_tracks=2, rng=rng)
        assert out is not None
        assert len(out.tracks) == 2
        for tr in out.tracks:
            assert len(tr.bars) == 4

    def test_select_window_all_tracks_share_same_bar_range(self):
        tr0_bars = [make_bar([make_note(pitch=60 + i)]) for i in range(8)]
        tr1_bars = [make_bar([make_note(pitch=70 + i)]) for i in range(8)]
        score = Score(tracks=[
            Track(bars=tr0_bars, track_type="melodic"),
            Track(bars=tr1_bars, track_type="melodic"),
        ], resolution=12)
        out = select_window(score, n_bars=4, n_tracks=2,
                            rng=random.Random(1))
        assert out is not None
        # Two tracks may emerge in either order; check that within each output
        # bar the two tracks' pitches differ by exactly 10 (proves alignment).
        diffs = []
        for i in range(4):
            p0 = out.tracks[0].bars[i].notes[0].pitch
            p1 = out.tracks[1].bars[i].notes[0].pitch
            diffs.append(abs(p1 - p0))
        assert diffs == [10, 10, 10, 10]

    def test_select_window_returns_none_when_score_too_short(self):
        score = Score(tracks=[melodic_track(n_bars=2)], resolution=12)
        out = select_window(score, n_bars=8, n_tracks=1,
                            rng=random.Random(0))
        assert out is None

    def test_select_window_deterministic_with_seeded_rng(self):
        score = Score(
            tracks=[melodic_track(n_bars=10), melodic_track(n_bars=10)],
            resolution=12,
        )
        a = select_window(score, n_bars=3, n_tracks=2,
                          rng=random.Random(123))
        b = select_window(score, n_bars=3, n_tracks=2,
                          rng=random.Random(123))
        assert a is not None and b is not None
        assert [n.pitch for tr in a.tracks for bar in tr.bars
                for n in bar.notes] == \
               [n.pitch for tr in b.tracks for bar in tr.bars
                for n in bar.notes]

    def test_select_window_does_not_mutate_source(self):
        score = Score(tracks=[melodic_track(n_bars=6)], resolution=12)
        before_pitches = _all_pitches(score)
        before_bar_count = _count_bars(score)
        _ = select_window(score, n_bars=3, n_tracks=1,
                          rng=random.Random(0))
        assert _all_pitches(score) == before_pitches
        assert _count_bars(score) == before_bar_count


# --------------------------------------------------------------------------- #
# MaskBar
# --------------------------------------------------------------------------- #
class TestMaskBar:
    def test_mask_bar_marks_some_bars_future(self):
        score = Score(tracks=[melodic_track(n_bars=8)], resolution=12)
        cfg = MaskBarConfig(apply_probability=1.0,
                            mode=MaskMode.RANDOM,
                            bar_fraction=0.5)
        random.seed(0)
        out = MaskBar(cfg)(copy.deepcopy(score))
        future_count = sum(1 for b in out.tracks[0].bars if b.future)
        assert future_count >= 1
        assert len(out.tracks[0].bars) == 8

    def test_mask_bar_apply_probability_zero_never_masks(self):
        score = Score(tracks=[melodic_track(n_bars=8)], resolution=12)
        cfg = MaskBarConfig(apply_probability=0.0)
        random.seed(0)
        out = MaskBar(cfg)(copy.deepcopy(score))
        assert not any(b.future for t in out.tracks for b in t.bars)

    def test_mask_bar_seeded_is_deterministic(self):
        score = Score(tracks=[melodic_track(n_bars=8)], resolution=12)
        cfg = MaskBarConfig(apply_probability=1.0,
                            mode=MaskMode.RANDOM,
                            bar_fraction=0.5)
        random.seed(123)
        a = MaskBar(cfg)(copy.deepcopy(score))
        random.seed(123)
        b = MaskBar(cfg)(copy.deepcopy(score))
        a_future = [b_.future for t in a.tracks for b_ in t.bars]
        b_future = [b_.future for t in b.tracks for b_ in t.bars]
        assert a_future == b_future

    def test_mask_bar_respects_infill_exclusion(self):
        score = Score(tracks=[melodic_track(n_bars=8)], resolution=12)
        cfg = MaskBarConfig(apply_probability=1.0,
                            mode=MaskMode.RANDOM,
                            bar_fraction=1.0)
        infill = {(0, 3)}
        for seed in range(20):
            random.seed(seed)
            out = MaskBar(cfg, infill_bars=infill)(copy.deepcopy(score))
            assert out.tracks[0].bars[3].future is False, \
                f"infill bar (0,3) was masked at seed={seed}"

    def test_mask_bar_preserves_note_and_bar_counts(self):
        score = Score(tracks=[melodic_track(n_bars=8)], resolution=12)
        cfg = MaskBarConfig(apply_probability=1.0, mode=MaskMode.STRUCTURED)
        nN, nB = _count_notes(score), _count_bars(score)
        random.seed(5)
        out = MaskBar(cfg)(copy.deepcopy(score))
        assert _count_notes(out) == nN
        assert _count_bars(out) == nB

    def test_mask_bar_probability_statistical(self):
        score = Score(tracks=[melodic_track(n_bars=6)], resolution=12)
        cfg = MaskBarConfig(apply_probability=0.5, mode=MaskMode.RANDOM,
                            bar_fraction=0.5)
        random.seed(0)
        masked_runs = 0
        N = 200
        for _ in range(N):
            out = MaskBar(cfg)(copy.deepcopy(score))
            if any(b.future for t in out.tracks for b in t.bars):
                masked_runs += 1
        assert 0.30 <= masked_runs / N <= 0.70


# --------------------------------------------------------------------------- #
# InstrumentSwap
# --------------------------------------------------------------------------- #
class TestInstrumentSwap:
    def test_instrument_swap_replaces_mapped_instrument(self):
        score = Score(
            tracks=[Track(bars=[make_bar()], instrument=0, track_type="melodic")],
            resolution=12,
        )
        random.seed(0)
        out = InstrumentSwap({0: [42]})(copy.deepcopy(score))
        assert out.tracks[0].instrument == 42

    def test_instrument_swap_leaves_unmapped_instrument_unchanged(self):
        score = Score(
            tracks=[Track(bars=[make_bar()], instrument=5, track_type="melodic")],
            resolution=12,
        )
        out = InstrumentSwap({0: [42]})(copy.deepcopy(score))
        assert out.tracks[0].instrument == 5

    def test_instrument_swap_is_seedable(self):
        score = Score(
            tracks=[Track(bars=[make_bar()], instrument=0, track_type="melodic")],
            resolution=12,
        )
        random.seed(7)
        a = InstrumentSwap({0: [1, 2, 3, 4, 5]})(copy.deepcopy(score))
        random.seed(7)
        b = InstrumentSwap({0: [1, 2, 3, 4, 5]})(copy.deepcopy(score))
        assert a.tracks[0].instrument == b.tracks[0].instrument

    def test_instrument_swap_preserves_note_count_and_track_count(self):
        score = Score(tracks=[melodic_track(n_bars=2),
                              melodic_track(n_bars=2)],
                      resolution=12)
        nN = _count_notes(score)
        out = InstrumentSwap({0: [10, 11]})(copy.deepcopy(score))
        assert len(out.tracks) == 2
        assert _count_notes(out) == nN


# --------------------------------------------------------------------------- #
# TrackPermutation
# --------------------------------------------------------------------------- #
class TestTrackPermutation:
    def test_track_permutation_preserves_track_count(self):
        score = Score(
            tracks=[melodic_track(n_bars=2, base_pitch=60),
                    melodic_track(n_bars=2, base_pitch=70),
                    drum_track(n_bars=2)],
            resolution=12,
        )
        random.seed(0)
        out = TrackPermutation()(copy.deepcopy(score))
        assert len(out.tracks) == 3

    def test_track_permutation_preserves_track_contents_set(self):
        score = Score(
            tracks=[melodic_track(n_bars=1, base_pitch=60),
                    melodic_track(n_bars=1, base_pitch=70),
                    melodic_track(n_bars=1, base_pitch=80)],
            resolution=12,
        )
        first_pitches_before = sorted(t.bars[0].notes[0].pitch
                                      for t in score.tracks)
        random.seed(0)
        out = TrackPermutation()(copy.deepcopy(score))
        first_pitches_after = sorted(t.bars[0].notes[0].pitch
                                     for t in out.tracks)
        assert first_pitches_before == first_pitches_after

    def test_track_permutation_is_seedable(self):
        tracks = [melodic_track(n_bars=1, base_pitch=60 + 5 * i)
                  for i in range(4)]
        score = Score(tracks=tracks, resolution=12)
        random.seed(11)
        a = TrackPermutation()(copy.deepcopy(score))
        random.seed(11)
        b = TrackPermutation()(copy.deepcopy(score))
        a_order = [t.bars[0].notes[0].pitch for t in a.tracks]
        b_order = [t.bars[0].notes[0].pitch for t in b.tracks]
        assert a_order == b_order

    def test_track_permutation_changes_order_for_some_seed(self):
        tracks = [melodic_track(n_bars=1, base_pitch=60 + 5 * i)
                  for i in range(6)]
        score = Score(tracks=tracks, resolution=12)
        original = [t.bars[0].notes[0].pitch for t in score.tracks]
        found_change = False
        for seed in range(50):
            random.seed(seed)
            out = TrackPermutation()(copy.deepcopy(score))
            shuffled = [t.bars[0].notes[0].pitch for t in out.tracks]
            if shuffled != original:
                found_change = True
                break
        assert found_change


# --------------------------------------------------------------------------- #
# AugmentationPipeline
# --------------------------------------------------------------------------- #
class TestAugmentationPipeline:
    def test_empty_pipeline_is_identity(self):
        score = Score(tracks=[melodic_track(n_bars=2)], resolution=12)
        out = AugmentationPipeline([])(score)
        assert _all_pitches(out)    == _all_pitches(score)
        assert _all_velocities(out) == _all_velocities(score)
        assert _count_bars(out)     == _count_bars(score)

    def test_pipeline_does_not_mutate_input(self):
        score = Score(tracks=[melodic_track(n_bars=2)], resolution=12)
        before_pitches = _all_pitches(score)
        before_vels    = _all_velocities(score)
        _ = AugmentationPipeline([Transpose(5), VelocityScale(0.5)])(score)
        # Input untouched (pipeline does deepcopy).
        assert _all_pitches(score)    == before_pitches
        assert _all_velocities(score) == before_vels

    def test_pipeline_composes_transforms_in_order(self):
        score = Score(tracks=[melodic_track(n_bars=2)], resolution=12)
        before_pitches = _all_pitches(score)
        before_vels    = _all_velocities(score)
        out = AugmentationPipeline([
            Transpose(5), VelocityScale(0.5),
        ])(score)
        assert _all_pitches(out)    == [max(0, min(127, p + 5))
                                        for p in before_pitches]
        assert _all_velocities(out) == [max(1, min(127, int(v * 0.5)))
                                        for v in before_vels]

    def test_pipeline_order_matters(self):
        # +20 then -20: 120 -> 127 (clamp) -> 107
        # -20 then +20: 120 ->  100        -> 120
        score = Score(tracks=[Track(bars=[make_bar([make_note(pitch=120, vel=80)])],
                                    track_type="melodic")], resolution=12)
        out_a = AugmentationPipeline([Transpose(20), Transpose(-20)])(
            copy.deepcopy(score))
        out_b = AugmentationPipeline([Transpose(-20), Transpose(20)])(
            copy.deepcopy(score))
        assert _all_pitches(out_a) == [107]
        assert _all_pitches(out_b) == [120]

    def test_default_training_pipeline_constructs_and_runs(self):
        pipeline = AugmentationPipeline.default_training()
        assert isinstance(pipeline, AugmentationPipeline)
        score = Score(tracks=[melodic_track(n_bars=2), drum_track(n_bars=2)],
                      resolution=12)
        random.seed(0)
        out = pipeline(score)
        assert isinstance(out, Score)
        assert len(out.tracks) == len(score.tracks)
        assert _count_bars(out)  == _count_bars(score)
        assert _count_notes(out) == _count_notes(score)


# --------------------------------------------------------------------------- #
# Documented gaps between TEST_IMPLEMENTATION_PLAN.md §3.8 and the source
# --------------------------------------------------------------------------- #
def test_plan_gap_velocity_class_named_velocity_scale():
    """Plan §3.8 calls the velocity-augmentation class `Velocity`, but the
    actual export is `VelocityScale`. This test documents the rename.
    """
    from midigpt.augmentation import VelocityScale as _VS
    assert _VS.__name__ == "VelocityScale"
    import midigpt.augmentation as aug
    assert not hasattr(aug, "Velocity"), (
        "Plan refers to `Velocity` but source only exports `VelocityScale`."
    )


def test_plan_gap_score_window_is_a_function_not_a_class():
    """Plan §3.8 mentions `ScoreWindow` as a transform class. The source
    actually exposes `select_window`, a function returning Optional[Score].
    """
    import midigpt.augmentation as aug
    assert callable(aug.select_window)
    assert not hasattr(aug, "ScoreWindow")


def test_plan_gap_bar_window_not_publicly_exported():
    """`BarWindow` exists in `midigpt.augmentation.bar_window` but is NOT
    re-exported via `midigpt.augmentation.__init__`. The submodule import
    path is the only public access today.
    """
    import midigpt.augmentation as aug
    assert not hasattr(aug, "BarWindow")
    from midigpt.augmentation.bar_window import BarWindow as _BW
    assert _BW.__name__ == "BarWindow"
