"""Validate a GenerationRequest against the encoder config and the score
*before* mask/planner/encode runs.

Layering:
  validate_request(request, score, encoder_config, analyzer=None) → request
       ↓
  StepPlanner → EncodeOptions per step
       ↓
  Encoder(score, opts) → tokens
"""

from __future__ import annotations

import json
import logging
from dataclasses import replace

from .config import GenerationRequest

log = logging.getLogger(__name__)


class RequestValidationError(ValueError):
    """Raised when a GenerationRequest is structurally invalid."""


# --- helpers ------------------------------------------------------------


def _config_dict(encoder_config) -> dict:
    try:
        return json.loads(encoder_config.to_json())
    except Exception:
        return {}


def _num_bars_values(cfg_dict: dict) -> list[int]:
    return list(cfg_dict.get("num_bars_map") or [])


def _time_signatures(cfg_dict: dict) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for entry in cfg_dict.get("time_signatures") or []:
        if isinstance(entry, str) and "/" in entry:
            try:
                n, d = entry.split("/")
                out.add((int(n), int(d)))
            except ValueError:
                continue
        elif isinstance(entry, list | tuple) and len(entry) == 2:
            out.add((int(entry[0]), int(entry[1])))
    return out


def _supports_mask_bar(cfg_dict: dict) -> bool:
    return any(d.get("type") == "MaskBar" for d in (cfg_dict.get("token_domains") or []))


def _attribute_control_names(encoder_config) -> set[str]:
    raw = getattr(encoder_config, "attribute_controls_json", "") or ""
    if not raw:
        return set()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return set()
    return {e.get("name") for e in parsed if isinstance(e, dict) and e.get("name")}


def _is_right_suffix(bars: list[int], total: int) -> bool:
    """True iff `bars` (as a set) equals {k, k+1, ..., total-1} for some k in [0, total]."""
    if not bars:
        return False
    s = set(bars)
    if len(s) != len(bars):
        return False  # duplicates
    if max(s) != total - 1:
        return False
    return s == set(range(total - len(s), total))


# --- main entry ---------------------------------------------------------


def validate_request(
    request: GenerationRequest, score, encoder_config, analyzer=None
) -> GenerationRequest:
    """Validate and fill defaults. Returns a (possibly new) request.

    Raises RequestValidationError on structural problems.
    Logs warnings for non-fatal data issues (out-of-range pitches, etc.).
    """
    cfg = request.config
    cfg_dict = _config_dict(encoder_config)

    # ---------------- model_dim default + domain ----------------
    nb_values = _num_bars_values(cfg_dict)
    if nb_values:
        if cfg.model_dim is None or cfg.model_dim <= 0:
            cfg = replace(cfg, model_dim=min(nb_values))
        elif cfg.model_dim not in nb_values:
            raise RequestValidationError(
                f"model_dim={cfg.model_dim} not in vocab domain {nb_values}"
            )

    # ---------------- sampling sanity ----------------
    if cfg.bars_per_step <= 0:
        raise RequestValidationError(f"bars_per_step must be > 0 (got {cfg.bars_per_step})")
    if cfg.bars_per_step > cfg.model_dim:
        raise RequestValidationError(
            f"bars_per_step={cfg.bars_per_step} cannot exceed model_dim={cfg.model_dim}"
        )
    if cfg.tracks_per_step <= 0:
        raise RequestValidationError(f"tracks_per_step must be > 0 (got {cfg.tracks_per_step})")
    if cfg.max_attempts < 1:
        raise RequestValidationError(f"max_attempts must be >= 1 (got {cfg.max_attempts})")
    if cfg.temperature <= 0:
        raise RequestValidationError(f"temperature must be > 0 (got {cfg.temperature})")
    # temperature_escalation: must be >= 1.0 (multiplier per failed attempt).
    # Cap at a reasonable maximum so retries can't explode into pure noise.
    _TEMP_ESC_MAX = 3.0
    if cfg.temperature_escalation < 1.0:
        raise RequestValidationError(
            f"temperature_escalation must be >= 1.0 (got {cfg.temperature_escalation}); "
            "use 1.0 to disable escalation"
        )
    if cfg.temperature_escalation > _TEMP_ESC_MAX:
        log.warning(
            "temperature_escalation=%.2f clamped to %.2f (max)",
            cfg.temperature_escalation,
            _TEMP_ESC_MAX,
        )
        cfg = replace(cfg, temperature_escalation=_TEMP_ESC_MAX)

    # ---------------- sampling filters ----------------
    # top_p ∈ (0, 1] : 1.0 = off (keep all). 0.0 would mean "keep nothing".
    if not (0.0 < cfg.top_p <= 1.0):
        raise RequestValidationError(
            f"top_p must be in (0, 1] (got {cfg.top_p}); use 1.0 to disable"
        )
    # mask_p ∈ [0, 1) : 0.0 = off. 1.0 would mask the entire distribution.
    if not (0.0 <= cfg.mask_p < 1.0):
        raise RequestValidationError(
            f"mask_p must be in [0, 1) (got {cfg.mask_p}); use 0.0 to disable"
        )
    if cfg.top_k < 0:
        raise RequestValidationError(f"top_k must be >= 0 (got {cfg.top_k}); use 0 to disable")
    if cfg.mask_k < 0:
        raise RequestValidationError(f"mask_k must be >= 0 (got {cfg.mask_k}); use 0 to disable")
    # Pool-emptiness guards (only when BOTH sides are active).
    if cfg.mask_p > 0.0 and cfg.top_p < 1.0 and cfg.mask_p >= cfg.top_p:
        raise RequestValidationError(
            f"mask_p ({cfg.mask_p}) must be < top_p ({cfg.top_p}); "
            f"mask_p chops the most-likely mass *within* the top_p nucleus, so "
            f"mask_p ≥ top_p empties the sampling pool"
        )
    if cfg.mask_k > 0 and cfg.top_k > 0 and cfg.mask_k >= cfg.top_k:
        raise RequestValidationError(
            f"mask_k ({cfg.mask_k}) must be < top_k ({cfg.top_k}); "
            f"mask_k removes the most-likely ranks *within* the top_k pool"
        )

    # ---------------- score shape ----------------
    if not score.tracks:
        raise RequestValidationError("score has 0 tracks")

    track_bar_counts = [len(t.bars) for t in score.tracks]
    if any(n == 0 for n in track_bar_counts):
        raise RequestValidationError(
            f"every track must have at least one bar (got lengths {track_bar_counts})"
        )
    if len(set(track_bar_counts)) > 1:
        raise RequestValidationError(
            f"all tracks must have the same number of bars (got {track_bar_counts})"
        )
    nb_score = track_bar_counts[0]
    if nb_score < cfg.model_dim:
        raise RequestValidationError(
            f"score has {nb_score} bars but model_dim={cfg.model_dim}; "
            f"model was trained on fixed windows ({nb_values or 'unknown'}). "
            "Pad the score or pick a smaller model_dim."
        )

    # ---------------- mask_mode vs encoder vocab ----------------
    # Bar masking is universal; the encoder chooses *how* to represent a
    # masked bar based on cfg.mask_mode. Only "token" mode needs the MaskBar
    # vocab entry — attention* / remove modes work on any encoder.
    mask_mode = getattr(cfg, "mask_mode", "token")
    if mask_mode == "token" and not _supports_mask_bar(cfg_dict):
        raise RequestValidationError(
            "config.mask_mode='token' requires the encoder vocab to include "
            "a MaskBar token domain (set supports_mask_bar_token=true in the "
            "encoder config, or pick mask_mode='attention'/'attention_approx'"
            "/'attention_skip'/'remove')."
        )

    # ---------------- time signatures ----------------
    allowed_ts = _time_signatures(cfg_dict)
    if allowed_ts:
        for ti, t in enumerate(score.tracks):
            for bi, b in enumerate(t.bars):
                tsn = getattr(b, "ts_numerator", None)
                tsd = getattr(b, "ts_denominator", None)
                if tsn and tsd and (tsn, tsd) not in allowed_ts:
                    raise RequestValidationError(
                        f"track {ti} bar {bi}: time signature {tsn}/{tsd} "
                        f"not in config.time_signatures"
                    )

    # ---------------- pitch range warning ----------------
    pmin = getattr(encoder_config, "pitch_min", 0)
    pmax = getattr(encoder_config, "pitch_max", 127)
    oor_count = 0
    for note in getattr(score, "notes", []):
        if note.pitch < pmin or note.pitch > pmax:
            oor_count += 1
    if oor_count:
        log.warning(
            "%d note(s) have pitches outside [%d, %d]; they will be dropped at encode time",
            oor_count,
            pmin,
            pmax,
        )

    # ---------------- instrument range warning ----------------
    # InstrumentGrouping covers all 128 GM programs by construction; anything
    # outside that range falls back to instrument 0 at encode time.
    for ti, t in enumerate(score.tracks):
        inst = getattr(t, "instrument", None)
        if inst is None:
            continue
        if not (0 <= inst <= 127):
            log.warning(
                "track %d: instrument %s out of GM range [0,127] — will fall back to 0 at encode time",
                ti,
                inst,
            )

    # ---------------- per-note structural warnings ----------------
    zero_dur = 0
    overflow_onset = 0
    ppq = int(getattr(score, "resolution", 480) or 480)
    for t in score.tracks:
        for b in t.bars:
            bar_notes = b.notes if hasattr(b, "notes") else [score.notes[i] for i in b.note_indices]
            for note in bar_notes:
                if note.duration_ticks <= 0:
                    zero_dur += 1
                tsn = getattr(b, "ts_numerator", 4) or 4
                tsd = getattr(b, "ts_denominator", 4) or 4
                bar_ticks = int(4 * ppq * tsn / tsd) if tsd else 0
                if bar_ticks > 0 and note.onset_ticks >= bar_ticks:
                    overflow_onset += 1
    if zero_dur:
        log.warning("%d note(s) have duration_ticks <= 0; they will be lost in roundtrip", zero_dur)
    if overflow_onset:
        log.warning(
            "%d note(s) have onset_ticks >= bar length; they will be dropped at encode time",
            overflow_onset,
        )

    # ---------------- per-track validation ----------------
    if not request.tracks:
        raise RequestValidationError("request.tracks is empty — nothing to generate")

    prompt_ids = {tp.id for tp in request.tracks}
    missing = sorted(set(range(len(score.tracks))) - prompt_ids)
    if missing:
        raise RequestValidationError(
            f"score has {len(score.tracks)} tracks but request is missing prompts "
            f"for track_id(s) {missing}; every score track requires an explicit "
            f"TrackPrompt (use ignore=True to skip)"
        )

    seen_ids: set[int] = set()
    has_infill = False
    has_anything_to_generate = False
    attr_sizes = analyzer.attribute_sizes() if analyzer is not None else {}
    # The analyzer exposes attributes under their *instance* names (e.g.
    # PolyphonyQuantile(mode="min") → "min_polyphony"), and that's the name the
    # rest of the pipeline keys on (encoder prompt overrides + constraint
    # builder). Fall back to the registry-key names from the encoder config
    # only when no analyzer is present.
    if attr_sizes:
        attr_names = set(attr_sizes.keys())
    else:
        attr_names = _attribute_control_names(encoder_config)
    attr_track_types = analyzer.attribute_track_types() if analyzer is not None else {}
    attr_levels = analyzer.attribute_levels() if analyzer is not None else {}
    # Non-attribute controls live on tp.controls. Each entry has its own
    # validator below; the names here are just the dispatch table.
    KNOWN_CONTROLS = {"time_signature"}
    _cfg_dict = _config_dict(encoder_config)
    _ts_count = len(_cfg_dict.get("time_signatures") or [])
    _ts_list = list(_cfg_dict.get("time_signatures") or [])

    # Cross-track time-signature coherence accumulator: bar_idx -> (ts_value, source_track_id).
    # Populated as each track's bar_controls is validated, then cross-checked
    # at the end of the per-track loop.
    ts_per_bar: dict[int, tuple[int, int]] = {}

    for tp in request.tracks:
        if tp.id in seen_ids:
            raise RequestValidationError(f"duplicate track_id={tp.id} in request")
        seen_ids.add(tp.id)

        if tp.id < 0 or tp.id >= len(score.tracks):
            raise RequestValidationError(
                f"track_id={tp.id} out of range (score has {len(score.tracks)} tracks)"
            )

        if tp.autoregressive and tp.ignore:
            raise RequestValidationError(
                f"track_id={tp.id}: autoregressive and ignore are mutually exclusive"
            )
        if tp.ignore and tp.bars:
            raise RequestValidationError(f"track_id={tp.id}: ignored tracks must not specify bars")

        nb_track = len(score.tracks[tp.id].bars)
        for b in tp.bars:
            if b < 0 or b >= nb_track:
                raise RequestValidationError(
                    f"track_id={tp.id}: bar {b} out of range (track has {nb_track} bars)"
                )

        # mask_bars validation: in-range, disjoint from `bars`, not on an
        # ignored track. Masking + generation cannot overlap — TOKEN_MASK_BAR
        # and the generation target encoding are mutually exclusive.
        if tp.mask_bars:
            if tp.ignore:
                raise RequestValidationError(
                    f"track_id={tp.id}: ignored tracks must not specify mask_bars"
                )
            for b in tp.mask_bars:
                if b < 0 or b >= nb_track:
                    raise RequestValidationError(
                        f"track_id={tp.id}: mask_bar {b} out of range (track has {nb_track} bars)"
                    )
            overlap = sorted(set(tp.mask_bars) & set(tp.bars))
            if overlap:
                raise RequestValidationError(
                    f"track_id={tp.id}: bars {overlap} appear in both `bars` "
                    f"(generation targets) and `mask_bars` — these states are "
                    f"mutually exclusive"
                )

        # AR bar-selection shape: must be a right-suffix of the track
        # (or the full track, which is a right-suffix with k=0).
        if tp.autoregressive and tp.bars:
            if not _is_right_suffix(tp.bars, nb_track):
                raise RequestValidationError(
                    f"track_id={tp.id} (autoregressive): bars must form a contiguous "
                    f"right-suffix of the track (e.g. [k, k+1, ..., {nb_track - 1}]); "
                    f"got {sorted(tp.bars)}"
                )

        # A bar cannot be both masked (future=True → MASK_BAR) and an infill
        # target (bars_to_generate → FillInPlaceholder/FillInStart/FillInEnd).
        # These states are mutually exclusive: the encoder resolves the conflict
        # silently (infill wins), so we catch it here instead.
        if not tp.autoregressive and not tp.ignore and tp.bars:
            masked_infill = [
                b for b in tp.bars if b < nb_track and score.tracks[tp.id].bars[b].future
            ]
            if masked_infill:
                raise RequestValidationError(
                    f"track_id={tp.id}: bars {masked_infill} are both marked as "
                    f"masked (future=True) and requested for infill generation — "
                    f"these states are mutually exclusive"
                )

        # An AR track's bars cannot be masked either (the model is supposed to
        # freely generate the whole track from the given context).
        if tp.autoregressive:
            masked_ar = [
                b
                for b in (tp.bars or range(nb_track))
                if b < nb_track and score.tracks[tp.id].bars[b].future
            ]
            if masked_ar:
                raise RequestValidationError(
                    f"track_id={tp.id} (autoregressive): bars {masked_ar} are "
                    f"marked as masked (future=True) — cannot mask bars on an "
                    f"autoregressive generation track"
                )

        # Selection accounting
        if tp.ignore:
            pass
        elif tp.autoregressive:
            has_anything_to_generate = True
        elif tp.bars:
            has_infill = True
            has_anything_to_generate = True

        # Attribute name + value range + track-type compatibility
        track = score.tracks[tp.id] if tp.id < len(score.tracks) else None
        is_drum_track = None
        if track is not None:
            tt = getattr(track, "track_type", None)
            if tt is None:
                from midigpt._core import TrackType

                tt = "drum" if track.type == TrackType.Drum else "melodic"
            is_drum_track = tt == "drum"

        for k, v in tp.attributes.items():
            if attr_names and k not in attr_names:
                raise RequestValidationError(
                    f"track_id={tp.id}: unknown attribute '{k}' "
                    f"(known: {sorted(attr_names)}); "
                    f"non-attribute controls (e.g. time_signature) live on tp.controls"
                )
            # Level-correctness: bar-level attributes must go in
            # `bar_attributes`, not `attributes`.
            if attr_levels.get(k) == "bar":
                raise RequestValidationError(
                    f"track_id={tp.id}: attribute '{k}' is bar-level — "
                    f"set it via tp.bar_attributes[bar_idx]['{k}'], "
                    f"not tp.attributes"
                )
            sz = attr_sizes.get(k)
            if sz is not None and not (0 <= int(v) < sz):
                raise RequestValidationError(
                    f"track_id={tp.id}: attribute '{k}'={v} out of range [0, {sz})"
                )
            req_tt = attr_track_types.get(k, "both")
            if is_drum_track is not None and req_tt != "both":
                if req_tt == "melodic" and is_drum_track:
                    raise RequestValidationError(
                        f"track_id={tp.id}: attribute '{k}' is melodic-only "
                        f"but track is a drum track"
                    )
                if req_tt == "drum" and not is_drum_track:
                    raise RequestValidationError(
                        f"track_id={tp.id}: attribute '{k}' is drum-only "
                        f"but track is a melodic track"
                    )
            # Achievability (warning only): in partial-AR / infill, a
            # track-level override is computed over the full track. Fixed
            # bars may already preclude the requested value.
            is_partial_ar = tp.autoregressive and tp.bars and min(tp.bars) > 0
            is_infill = not tp.autoregressive and tp.bars
            if (is_partial_ar or is_infill) and analyzer is not None:
                attr_obj = analyzer.get(k) if hasattr(analyzer, "get") else None
                if attr_obj is not None:
                    try:
                        lo, hi = attr_obj.achievable_range(score, tp.id, list(tp.bars))
                    except Exception:
                        lo, hi = (0, (sz or 1) - 1)
                    if not (lo <= int(v) <= hi):
                        log.warning(
                            "track_id=%d: attribute '%s'=%d is outside "
                            "achievable range [%d, %d] given the fixed "
                            "bars — request accepted, but the realized "
                            "attribute will not match (try a value in range, "
                            "or switch to full AR to regenerate the whole "
                            "track)",
                            tp.id,
                            k,
                            int(v),
                            lo,
                            hi,
                        )

        # ---------- per-bar attribute overrides ----------
        gen_bars_set = set(tp.bars or [])
        # Suffix-only rule for partial-AR is structurally equivalent to
        # "bar_idx ∈ tp.bars" (the suffix). For full-AR with no prefix,
        # tp.bars may be empty meaning "the whole track"; in that case any
        # in-range bar idx is allowed.
        full_track_ar = tp.autoregressive and not tp.bars
        nb_track_ref = nb_track  # for use in lambdas below
        for bar_idx, bar_dict in (tp.bar_attributes or {}).items():
            bar_idx_i = int(bar_idx)
            if bar_idx_i < 0 or bar_idx_i >= nb_track_ref:
                raise RequestValidationError(
                    f"track_id={tp.id}: bar_attributes key {bar_idx_i} out "
                    f"of range (track has {nb_track_ref} bars)"
                )
            if not full_track_ar and bar_idx_i not in gen_bars_set:
                raise RequestValidationError(
                    f"track_id={tp.id}: bar_attributes references bar "
                    f"{bar_idx_i}, which is not in tp.bars "
                    f"(generation targets); per-bar overrides must apply to "
                    f"bars being generated"
                )
            for k, v in (bar_dict or {}).items():
                if attr_names and k not in attr_names:
                    raise RequestValidationError(
                        f"track_id={tp.id} bar {bar_idx_i}: unknown attribute "
                        f"'{k}' (known: {sorted(attr_names)})"
                    )
                if attr_levels.get(k, "track") != "bar":
                    raise RequestValidationError(
                        f"track_id={tp.id} bar {bar_idx_i}: attribute '{k}' "
                        f"is track-level — set it via tp.attributes, not "
                        f"tp.bar_attributes"
                    )
                sz = attr_sizes.get(k)
                if sz is not None and not (0 <= int(v) < sz):
                    raise RequestValidationError(
                        f"track_id={tp.id} bar {bar_idx_i}: attribute '{k}'"
                        f"={v} out of range [0, {sz})"
                    )
                req_tt = attr_track_types.get(k, "both")
                if is_drum_track is not None and req_tt != "both":
                    if req_tt == "melodic" and is_drum_track:
                        raise RequestValidationError(
                            f"track_id={tp.id} bar {bar_idx_i}: attribute "
                            f"'{k}' is melodic-only but track is a drum track"
                        )
                    if req_tt == "drum" and not is_drum_track:
                        raise RequestValidationError(
                            f"track_id={tp.id} bar {bar_idx_i}: attribute "
                            f"'{k}' is drum-only but track is a melodic track"
                        )

        # ---------- per-bar non-attribute controls ----------
        for bar_idx, bar_dict in (tp.bar_controls or {}).items():
            bar_idx_i = int(bar_idx)
            if bar_idx_i < 0 or bar_idx_i >= nb_track_ref:
                raise RequestValidationError(
                    f"track_id={tp.id}: bar_controls key {bar_idx_i} out "
                    f"of range (track has {nb_track_ref} bars)"
                )
            if not full_track_ar and bar_idx_i not in gen_bars_set:
                raise RequestValidationError(
                    f"track_id={tp.id}: bar_controls references bar "
                    f"{bar_idx_i}, which is not in tp.bars"
                )
            for k, v in (bar_dict or {}).items():
                if k not in KNOWN_CONTROLS:
                    raise RequestValidationError(
                        f"track_id={tp.id} bar {bar_idx_i}: unknown control "
                        f"'{k}' (known: {sorted(KNOWN_CONTROLS)})"
                    )
                if k == "time_signature":
                    if _ts_count == 0:
                        raise RequestValidationError(
                            f"track_id={tp.id} bar {bar_idx_i}: encoder has "
                            f"no time_signatures configured"
                        )
                    if not (0 <= int(v) < _ts_count):
                        raise RequestValidationError(
                            f"track_id={tp.id} bar {bar_idx_i}: "
                            f"time_signature index {v} out of range "
                            f"[0, {_ts_count})"
                        )
                    # Cross-track coherence: same bar_idx must agree across
                    # generating tracks.
                    if bar_idx_i in ts_per_bar:
                        prev_v, prev_tid = ts_per_bar[bar_idx_i]
                        if prev_v != int(v):
                            raise RequestValidationError(
                                f"bar {bar_idx_i}: time_signature mismatch "
                                f"between track_id={prev_tid} (={prev_v}) "
                                f"and track_id={tp.id} (={int(v)}); "
                                f"a bar must have one time signature across "
                                f"all tracks"
                            )
                    else:
                        ts_per_bar[bar_idx_i] = (int(v), tp.id)
                    # Cross-check against any context-bar TS already in the
                    # score: every track has a bar at this index, and they
                    # must already agree (validated earlier in the score
                    # shape block). Just check track 0's bar TS for this
                    # index against the override.
                    if _ts_list and tp.id < len(score.tracks):
                        sb = score.tracks[tp.id].bars[bar_idx_i]
                        tsn = getattr(sb, "ts_numerator", None)
                        tsd = getattr(sb, "ts_denominator", None)
                        entry = _ts_list[int(v)] if int(v) < len(_ts_list) else None
                        if tsn and tsd and entry:
                            if isinstance(entry, str) and "/" in entry:
                                en, ed = entry.split("/")
                                en, ed = int(en), int(ed)
                            elif isinstance(entry, list | tuple) and len(entry) == 2:
                                en, ed = int(entry[0]), int(entry[1])
                            else:
                                en = ed = None
                            if en and ed and (en, ed) != (tsn, tsd):
                                raise RequestValidationError(
                                    f"track_id={tp.id} bar {bar_idx_i}: "
                                    f"time_signature override {en}/{ed} "
                                    f"conflicts with existing bar TS "
                                    f"{tsn}/{tsd}"
                                )

        # Non-attribute controls (tp.controls). Each has its own validator.
        controls = getattr(tp, "controls", {}) or {}
        for k, v in controls.items():
            if k not in KNOWN_CONTROLS:
                raise RequestValidationError(
                    f"track_id={tp.id}: unknown control '{k}' (known: {sorted(KNOWN_CONTROLS)})"
                )
            if k == "time_signature":
                if _ts_count == 0:
                    raise RequestValidationError(
                        f"track_id={tp.id}: encoder has no time_signatures "
                        f"configured — cannot lock time_signature"
                    )
                if not (0 <= int(v) < _ts_count):
                    raise RequestValidationError(
                        f"track_id={tp.id}: time_signature index {v} out of range [0, {_ts_count})"
                    )

    if not has_anything_to_generate:
        raise RequestValidationError(
            "request has no tracks to generate (every track is ignored or has no bars)"
        )

    if has_infill and not getattr(encoder_config, "supports_infill", False):
        raise RequestValidationError(
            "request requires infill but encoder config has supports_infill=false"
        )

    return GenerationRequest(tracks=request.tracks, config=cfg)
