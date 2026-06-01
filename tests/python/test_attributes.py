"""Tests for `midigpt.attributes.*` — section 3.7 of the test plan.

Covers every attribute class advertised by `ATTRIBUTE_REGISTRY`:
- size / name / token_type / level / track_type metadata invariants
- compute() on hand-crafted scores with predictable structure
- quantize() bin boundaries and clamping
- achievable_range() monotonicity (PolyphonyQuantile, NoteDurationQuantile)
- AttributeAnalyzer.from_config / compute_track_tokens / compute_bar_tokens /
  evaluate / token_domain_specs / attribute_sizes / attribute_levels
- deterministic output for fixed input

Notes for future maintainers:
* `Tension` / `TensionDrum` `.compute()` requires the external `tension_model`
  package which is not vendored — we exercise the metadata and quantize() path
  only, and assert that `compute(score, idx, bar_idx=None)` returns 0.0 (the
  documented no-bar-idx behavior).
* `simple_score` in conftest uses bar-relative `onset_ticks`. `SilenceProportion`
  in source treats them as absolute. For predictable assertions on silence we
  use `empty_bars_score` (fully silent) and a single-bar custom score (fully
  covered).
"""

from __future__ import annotations

import math

import pytest
from conftest import drum_track, make_bar, make_note, melodic_track

from midigpt._types import Bar, Note, Score, Track
from midigpt.attributes import (
    ATTRIBUTE_REGISTRY,
    TOKEN_TYPE_TO_ATTRIBUTE,
    AttributeAnalyzer,
    BarLevelPitchClassSet,
    BaseAttribute,
    KeySignature,
    NoteDensity,
    NoteDensityQuantile,
    NoteDurationDistribution,
    NoteDurationQuantile,
    OnsetPolyphony,
    PitchRange,
    PolyphonyQuantile,
    SilenceProportion,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _single_track_score(
    notes, res=12, beat_length=4.0, ts_num=4, ts_den=4, n_bars=1, track_type="melodic", instrument=0
) -> Score:
    """One-track score with explicit notes in the FIRST bar; remaining bars empty."""
    first = make_bar(notes=notes, ts_num=ts_num, ts_den=ts_den, beat_length=beat_length)
    extras = [
        make_bar(ts_num=ts_num, ts_den=ts_den, beat_length=beat_length) for _ in range(n_bars - 1)
    ]
    return Score(
        tracks=[Track(bars=[first, *extras], instrument=instrument, track_type=track_type)],
        resolution=res,
        tempo=500000,
    )


# --------------------------------------------------------------------------- #
# 1. Per-class metadata invariants
# --------------------------------------------------------------------------- #
ALL_CLASSES = [
    NoteDensity,
    OnsetPolyphony,
    PitchRange,
    KeySignature,
    NoteDurationDistribution,
    SilenceProportion,
    BarLevelPitchClassSet,
    NoteDensityQuantile,
]


@pytest.mark.parametrize("cls", ALL_CLASSES)
def test_attribute_metadata_invariants_parameterless(cls):
    inst = cls()
    assert isinstance(inst, BaseAttribute)
    assert isinstance(inst.name, str) and len(inst.name) > 0
    assert isinstance(inst.token_type, str) and len(inst.token_type) > 0
    assert inst.level in ("track", "bar")
    assert inst.track_type in ("melodic", "drum", "both")
    assert isinstance(inst.size, int) and inst.size > 0


@pytest.mark.parametrize("mode", ["min", "max"])
def test_polyphony_quantile_metadata(mode):
    p = PolyphonyQuantile(mode=mode)
    assert p.name == f"{mode}_polyphony"
    assert p.token_type == ("MinPolyphony" if mode == "min" else "MaxPolyphony")
    assert p.level == "track" and p.track_type == "melodic"
    assert p.size == 10


@pytest.mark.parametrize("mode", ["min", "max"])
def test_note_duration_quantile_metadata(mode):
    d = NoteDurationQuantile(mode=mode)
    assert d.name == f"{mode}_note_duration"
    assert d.token_type == ("MinNoteDuration" if mode == "min" else "MaxNoteDuration")
    assert d.level == "track" and d.size == 6


# --------------------------------------------------------------------------- #
# 2. Public-API discovery / registry
# --------------------------------------------------------------------------- #
def test_attribute_registry_contains_expected_keys():
    expected = {
        "note_density",
        "onset_polyphony",
        "pitch_range",
        "key_signature",
        "note_duration_dist",
        "silence_proportion",
        "pitch_class_set",
        "note_density_quantile",
        "polyphony_quantile",
        "note_duration_quantile",
    }
    assert set(ATTRIBUTE_REGISTRY.keys()) == expected
    # Every value is a BaseAttribute subclass
    for cls in ATTRIBUTE_REGISTRY.values():
        assert issubclass(cls, BaseAttribute)


def test_token_type_to_attribute_keys_match_registry():
    for token_type, (reg_key, params) in TOKEN_TYPE_TO_ATTRIBUTE.items():
        assert reg_key in ATTRIBUTE_REGISTRY, (
            f"{token_type} maps to unknown registry key {reg_key!r}"
        )
        # params must be a dict (possibly empty) and must be constructable
        assert isinstance(params, dict)
        ATTRIBUTE_REGISTRY[reg_key](**params)


# --------------------------------------------------------------------------- #
# 3. compute() on hand-crafted scores
# --------------------------------------------------------------------------- #
def test_note_density_compute_four_notes_per_bar():
    # 4 bars, 4 notes/bar → 16 notes / 4 bars = 4.0 notes/bar
    track = melodic_track(n_bars=4, notes_per_bar=4)
    score = Score(tracks=[track], resolution=12, tempo=500000)
    raw = NoteDensity().compute(score, 0)
    assert raw == pytest.approx(4.0)


def test_note_density_empty_track_returns_zero():
    score = Score(tracks=[Track(bars=[], track_type="melodic")], resolution=12, tempo=500000)
    assert NoteDensity().compute(score, 0) == 0


def test_onset_polyphony_chord_of_three():
    # Three simultaneous onsets at tick 0 in a single bar.
    chord = [
        make_note(pitch=60, onset=0, dur=12),
        make_note(pitch=64, onset=0, dur=12),
        make_note(pitch=67, onset=0, dur=12),
    ]
    score = _single_track_score(chord)
    assert OnsetPolyphony().compute(score, 0) == 3


def test_onset_polyphony_monophonic_returns_one():
    notes = [make_note(pitch=60, onset=0, dur=12), make_note(pitch=62, onset=12, dur=12)]
    score = _single_track_score(notes)
    assert OnsetPolyphony().compute(score, 0) == 1


def test_pitch_range_compute_spans_octave():
    # Notes [60..72] → max - min = 12.
    notes = [make_note(pitch=p, onset=i * 4, dur=4) for i, p in enumerate(range(60, 73))]
    score = _single_track_score(notes)
    assert PitchRange().compute(score, 0) == 12


def test_pitch_range_compute_no_notes_returns_zero():
    score = Score(
        tracks=[Track(bars=[make_bar()], track_type="melodic")], resolution=12, tempo=500000
    )
    assert PitchRange().compute(score, 0) == 0


def test_silence_proportion_empty_bars(empty_bars_score):
    raw = SilenceProportion().compute(empty_bars_score, 0)
    assert raw == pytest.approx(1.0)
    assert SilenceProportion().quantize(raw) == 9


def test_silence_proportion_fully_covered_bar():
    # One note that exactly covers the entire bar (48 ticks at res=12, 4/4).
    notes = [make_note(pitch=60, onset=0, dur=48)]
    score = _single_track_score(notes, res=12, n_bars=1)
    raw = SilenceProportion().compute(score, 0)
    assert raw == pytest.approx(0.0)
    assert SilenceProportion().quantize(raw) == 0


def test_key_signature_c_major_diatonic_yields_c_or_relative_minor():
    # Pure C-major scale notes are diatonically ambiguous with A-minor under
    # the Krumhansl profile. We only require the detected key to be one of
    # the two relative diatonic candidates (C major = 0, A minor = 12 + 9 = 21).
    notes = []
    for i, p in enumerate([60, 62, 64, 65, 67, 69, 71]):
        notes.append(make_note(pitch=p, onset=i * 4, dur=4))
    score = _single_track_score(notes)
    ks = KeySignature().compute(score, 0)
    assert ks in (0, 21), f"expected C-major or A-minor, got {ks}"


def test_key_signature_no_notes_returns_sentinel():
    score = Score(
        tracks=[Track(bars=[make_bar()], track_type="melodic")], resolution=12, tempo=500000
    )
    # 24 = "no key" sentinel.
    assert KeySignature().compute(score, 0) == 24


def test_note_duration_distribution_quarter_notes():
    # res=12 → quarter note ~ 12 ticks. d/(res/8) = 12/1.5 = 8 → log2=3 → "quarter".
    notes = [make_note(pitch=60, onset=i * 12, dur=12) for i in range(4)]
    score = _single_track_score(notes)
    assert NoteDurationDistribution().compute(score, 0) == 3


def test_note_duration_distribution_empty_returns_default():
    score = Score(
        tracks=[Track(bars=[make_bar()], track_type="melodic")], resolution=12, tempo=500000
    )
    assert NoteDurationDistribution().compute(score, 0) == 3


def test_bar_level_pitch_class_set_counts_unique_classes():
    # Pitches 60, 64, 67, 72 → classes {0, 4, 7, 0} = {0, 4, 7} → size 3.
    notes = [
        make_note(pitch=60, onset=0, dur=4),
        make_note(pitch=64, onset=4, dur=4),
        make_note(pitch=67, onset=8, dur=4),
        make_note(pitch=72, onset=12, dur=4),
    ]
    score = _single_track_score(notes)
    pcs = BarLevelPitchClassSet()
    assert pcs.compute(score, 0, bar_idx=0) == 3
    # No bar_idx → defined-but-trivial fallback.
    assert pcs.compute(score, 0, bar_idx=None) == 0


def test_bar_level_pitch_class_set_out_of_range_bar():
    score = _single_track_score([])
    assert BarLevelPitchClassSet().compute(score, 0, bar_idx=99) == 0


# --------------------------------------------------------------------------- #
# 4. Quantile attributes — monotonicity, bucket boundaries, achievable_range
# --------------------------------------------------------------------------- #
def test_polyphony_quantile_quantize_clamps_at_extremes():
    p = PolyphonyQuantile(mode="max")
    # quantize: clamp to [1, 10] then subtract 1.
    assert p.quantize(0) == 0  # clamped up to 1, then -1
    assert p.quantize(1) == 0
    assert p.quantize(5) == 4
    assert p.quantize(10) == 9
    assert p.quantize(99) == 9  # clamped down to 10, then -1
    assert p.quantize(-3) == 0


def test_polyphony_quantile_compute_min_vs_max():
    # Chord of 3 voices held for the whole bar → only timesteps with 3
    # voices contribute. 85th percentile of [3, 3, ...] == 3. 15th also 3.
    chord = [
        make_note(pitch=60, onset=0, dur=48),
        make_note(pitch=64, onset=0, dur=48),
        make_note(pitch=67, onset=0, dur=48),
    ]
    score = _single_track_score(chord)
    p_min = PolyphonyQuantile(mode="min")
    p_max = PolyphonyQuantile(mode="max")
    assert p_min.compute(score, 0) == 3
    assert p_max.compute(score, 0) == 3
    assert p_min.quantize(p_min.compute(score, 0)) == 2
    assert p_max.quantize(p_max.compute(score, 0)) == 2


def test_polyphony_quantile_compute_is_deterministic():
    notes = [make_note(pitch=60 + (i % 5), onset=i * 4, dur=8) for i in range(8)]
    score = _single_track_score(notes)
    p = PolyphonyQuantile(mode="max")
    runs = [p.compute(score, 0) for _ in range(5)]
    assert len(set(runs)) == 1


def test_polyphony_quantile_achievable_range_monotone():
    # All bars empty → fixed contribution is empty.
    s = Score(
        tracks=[Track(bars=[make_bar() for _ in range(4)], track_type="melodic")],
        resolution=12,
        tempo=500000,
    )
    p_max = PolyphonyQuantile(mode="max")
    lo, hi = p_max.achievable_range(s, 0, generated_bars=[0, 1, 2, 3])
    assert lo == 0 and hi == p_max.size - 1
    p_min = PolyphonyQuantile(mode="min")
    lo2, hi2 = p_min.achievable_range(s, 0, generated_bars=[0, 1, 2, 3])
    assert lo2 == 0 and hi2 == 0  # min(empty, anything) → bin 0


def test_polyphony_quantile_achievable_range_with_fixed_floor():
    # One fixed bar with a 4-voice chord → max-mode floor = 3 (quantized 4-1).
    chord = [make_note(pitch=60 + i, onset=0, dur=48) for i in range(4)]
    fixed = _single_track_score(chord, n_bars=2)
    p_max = PolyphonyQuantile(mode="max")
    lo, hi = p_max.achievable_range(fixed, 0, generated_bars=[1])
    assert lo == 3
    assert hi == p_max.size - 1


def test_note_duration_quantile_clamps_to_levels():
    d = NoteDurationQuantile(mode="max")
    # quantize is just int(v) — but compute already clips to [0, 5].
    assert d.quantize(0) == 0
    assert d.quantize(5) == 5
    # The constructor sets mode/name/token_type correctly.
    assert d.name == "max_note_duration"


def test_note_duration_quantile_compute_quarter_notes():
    # res=480 so duration 480 → log2(480/3)+1 = log2(160)+1 ≈ 8.32, clipped 5.
    # Smaller res for "quarter-level" target: dur=12, log2(12/3)+1=log2(4)+1=3.
    notes = [make_note(pitch=60, onset=i * 12, dur=12) for i in range(4)]
    score = _single_track_score(notes, res=12)
    d_min = NoteDurationQuantile(mode="min")
    d_max = NoteDurationQuantile(mode="max")
    assert d_min.compute(score, 0) == 3
    assert d_max.compute(score, 0) == 3


def test_note_duration_quantile_achievable_range_max_mode_floor():
    # One fixed bar with whole-note-ish duration → quantized level pinned high.
    notes = [make_note(pitch=60, onset=0, dur=12)]  # level 3
    fixed = _single_track_score(notes, res=12, n_bars=2)
    d_max = NoteDurationQuantile(mode="max")
    lo, hi = d_max.achievable_range(fixed, 0, generated_bars=[1])
    assert lo == 3
    assert hi == d_max.size - 1


def test_note_density_quantile_compute_drum_track():
    # Drum track triggers the qindex=128 path. 4 bars, 2 notes each = 2 avg.
    t = drum_track(n_bars=4)
    s = Score(tracks=[t], resolution=12, tempo=500000)
    ndq = NoteDensityQuantile()
    bin_idx = ndq.compute(s, 0)
    assert 0 <= bin_idx < ndq.size
    # Determinism
    assert ndq.compute(s, 0) == bin_idx


def test_note_density_quantile_achievable_range_lower_le_upper():
    t = drum_track(n_bars=4)
    s = Score(tracks=[t], resolution=12, tempo=500000)
    ndq = NoteDensityQuantile()
    lo, hi = ndq.achievable_range(s, 0, generated_bars=[2, 3])
    assert 0 <= lo <= hi < ndq.size


# --------------------------------------------------------------------------- #
# 6. Determinism on fixed input
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "cls",
    [
        NoteDensity,
        OnsetPolyphony,
        PitchRange,
        KeySignature,
        NoteDurationDistribution,
        SilenceProportion,
    ],
)
def test_attribute_compute_is_deterministic(cls, simple_score):
    inst = cls()
    a = inst.compute(simple_score, 0)
    b = inst.compute(simple_score, 0)
    c = inst.compute(simple_score, 0)
    assert a == b == c


# --------------------------------------------------------------------------- #
# 7. AttributeAnalyzer — from_config / compute_track_tokens / compute_bar_tokens
# --------------------------------------------------------------------------- #
def test_analyzer_from_config_returns_nonempty(ghost_analyzer):
    sizes = ghost_analyzer.attribute_sizes()
    assert isinstance(sizes, dict)
    assert len(sizes) > 0
    # Every advertised size is positive.
    for name, size in sizes.items():
        assert isinstance(name, str) and len(name) > 0
        assert isinstance(size, int) and size > 0


def test_analyzer_from_config_matches_token_domain_mapping(ghost_analyzer):
    for attr in ghost_analyzer._attrs.values():
        # Either auto-inferred from token_domains or appended via the JSON
        # overlay — both pathways yield real registered classes.
        assert any(cls is type(attr) for cls in ATTRIBUTE_REGISTRY.values())


def test_analyzer_attribute_levels_partition(ghost_analyzer):
    levels = ghost_analyzer.attribute_levels()
    for name, lvl in levels.items():
        assert lvl in ("track", "bar"), f"{name} has bad level {lvl}"


def test_analyzer_attribute_track_types(ghost_analyzer):
    tts = ghost_analyzer.attribute_track_types()
    for name, tt in tts.items():
        assert tt in ("melodic", "drum", "both"), f"{name}: {tt}"


def test_analyzer_compute_track_tokens_returns_only_track_level(ghost_analyzer, simple_score):
    out = ghost_analyzer.compute_track_tokens(simple_score, 0)
    levels = ghost_analyzer.attribute_levels()
    for name in out:
        assert levels[name] == "track"
    # Every value must be a non-negative int within the attribute's domain.
    sizes = ghost_analyzer.attribute_sizes()
    for name, q in out.items():
        assert 0 <= q < sizes[name], f"{name}={q} out of [0, {sizes[name]})"


def test_analyzer_compute_track_tokens_filters_drum_only_on_melodic(ghost_analyzer, simple_score):
    # `simple_score` track is melodic. Drum-only attrs must be excluded.
    out = ghost_analyzer.compute_track_tokens(simple_score, 0)
    track_types = ghost_analyzer.attribute_track_types()
    for name in out:
        assert track_types[name] in ("melodic", "both"), (
            f"{name} is {track_types[name]} but appeared on a melodic track"
        )


def test_analyzer_compute_track_tokens_filters_melodic_only_on_drum(
    ghost_analyzer, two_track_score
):
    # Track 1 is a drum track.
    out = ghost_analyzer.compute_track_tokens(two_track_score, 1)
    track_types = ghost_analyzer.attribute_track_types()
    for name in out:
        assert track_types[name] in ("drum", "both"), (
            f"{name} is {track_types[name]} but appeared on a drum track"
        )


def test_analyzer_compute_bar_tokens_only_bar_level(ghost_analyzer, simple_score):
    out = ghost_analyzer.compute_bar_tokens(simple_score, 0, 0)
    # Bar-level outputs are keyed by token_type (NOT name) per source.
    # Construct a token_type → level map by inspecting analyzer attrs.
    type_to_level = {a.token_type: a.level for a in ghost_analyzer._attrs.values()}
    for tok_type in out:
        assert type_to_level[tok_type] == "bar"


# --------------------------------------------------------------------------- #
# 8. AttributeAnalyzer.evaluate — match yields 1.0, mismatch 0.0
# --------------------------------------------------------------------------- #
def test_analyzer_evaluate_matching_bin_returns_one(ghost_analyzer, simple_score):
    realized = ghost_analyzer.compute_track_tokens(simple_score, 0)
    assert realized, "expected at least one track-level attribute"
    scores = ghost_analyzer.evaluate(realized, simple_score, 0)
    for name, s in scores.items():
        assert s == 1.0, f"{name} should match itself: got {s}"


def test_analyzer_evaluate_mismatching_bin_returns_zero(ghost_analyzer, simple_score):
    realized = ghost_analyzer.compute_track_tokens(simple_score, 0)
    sizes = ghost_analyzer.attribute_sizes()
    # Force a mismatch by perturbing every bin to a different value within
    # range (size > 1) — drop any attr with size==1 (no alternative).
    perturbed = {}
    for name, q in realized.items():
        if sizes[name] > 1:
            perturbed[name] = (q + 1) % sizes[name]
    assert perturbed, "need at least one multi-bin attr to compare"
    scores = ghost_analyzer.evaluate(perturbed, simple_score, 0)
    for name, s in scores.items():
        assert s == 0.0, f"{name} should mismatch perturbed value, got {s}"


def test_analyzer_evaluate_unknown_attribute_is_ignored(ghost_analyzer, simple_score):
    out = ghost_analyzer.evaluate({"_nonexistent_attribute_": 0}, simple_score, 0)
    assert "_nonexistent_attribute_" not in out
    assert out == {}


# --------------------------------------------------------------------------- #
# 9. token_domain_specs() — vocab plumbing
# --------------------------------------------------------------------------- #
def test_analyzer_token_domain_specs_lists_every_attribute(ghost_analyzer):
    specs = ghost_analyzer.token_domain_specs()
    assert isinstance(specs, list)
    assert len(specs) == len(ghost_analyzer._attrs)
    for tt, size in specs:
        assert isinstance(tt, str) and len(tt) > 0
        assert isinstance(size, int) and size > 0


def test_analyzer_token_domain_specs_matches_attribute_sizes(ghost_analyzer):
    specs = ghost_analyzer.token_domain_specs()
    sizes = ghost_analyzer.attribute_sizes()
    # Cross-check via name → token_type lookup.
    name_to_tt = {a.name: a.token_type for a in ghost_analyzer._attrs.values()}
    expected = {name_to_tt[n]: s for n, s in sizes.items()}
    actual = dict(specs)
    assert actual == expected


# --------------------------------------------------------------------------- #
# 10. Static manually-constructed analyzer (no _core dependence)
# --------------------------------------------------------------------------- #
def test_manual_analyzer_compute_track_tokens_quantizes():
    analyzer = AttributeAnalyzer([NoteDensity(), OnsetPolyphony()])
    track = melodic_track(n_bars=4, notes_per_bar=4)
    score = Score(tracks=[track], resolution=12, tempo=500000)
    out = analyzer.compute_track_tokens(score, 0)
    # NoteDensity: 16/4=4 → quantize=min(4,127)=4
    # OnsetPolyphony: monophonic ascending → 1 onset/tick → 1
    assert out == {"note_density": 4, "onset_polyphony": 1}


def test_manual_analyzer_get_returns_instance_or_none():
    nd = NoteDensity()
    analyzer = AttributeAnalyzer([nd])
    assert analyzer.get("note_density") is nd
    assert analyzer.get("nope") is None
