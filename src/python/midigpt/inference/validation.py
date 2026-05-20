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
        elif isinstance(entry, (list, tuple)) and len(entry) == 2:
            out.add((int(entry[0]), int(entry[1])))
    return out


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

def validate_request(request: GenerationRequest, score, encoder_config,
                     analyzer=None) -> GenerationRequest:
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
            cfg.temperature_escalation, _TEMP_ESC_MAX,
        )
        cfg = replace(cfg, temperature_escalation=_TEMP_ESC_MAX)

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
            oor_count, pmin, pmax,
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
                ti, inst,
            )

    # ---------------- per-note structural warnings ----------------
    zero_dur = 0
    overflow_onset = 0
    ppq = int(getattr(score, "resolution", 480) or 480)
    for t in score.tracks:
        for b in t.bars:
            for note in b.notes:
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

    seen_ids: set[int] = set()
    has_infill = False
    has_anything_to_generate = False
    attr_names = _attribute_control_names(encoder_config)
    attr_sizes = analyzer.attribute_sizes() if analyzer is not None else {}

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
            raise RequestValidationError(
                f"track_id={tp.id}: ignored tracks must not specify bars"
            )

        nb_track = len(score.tracks[tp.id].bars)
        for b in tp.bars:
            if b < 0 or b >= nb_track:
                raise RequestValidationError(
                    f"track_id={tp.id}: bar {b} out of range (track has {nb_track} bars)"
                )

        # AR bar-selection shape: must be a right-suffix of the track
        # (or the full track, which is a right-suffix with k=0).
        if tp.autoregressive and tp.bars:
            if not _is_right_suffix(tp.bars, nb_track):
                raise RequestValidationError(
                    f"track_id={tp.id} (autoregressive): bars must form a contiguous "
                    f"right-suffix of the track (e.g. [k, k+1, ..., {nb_track-1}]); "
                    f"got {sorted(tp.bars)}"
                )

        # A bar cannot be both masked (future=True → MASK_BAR) and an infill
        # target (bars_to_generate → FillInPlaceholder/FillInStart/FillInEnd).
        # These states are mutually exclusive: the encoder resolves the conflict
        # silently (infill wins), so we catch it here instead.
        if not tp.autoregressive and not tp.ignore and tp.bars:
            masked_infill = [
                b for b in tp.bars
                if b < nb_track and score.tracks[tp.id].bars[b].future
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
                b for b in (tp.bars or range(nb_track))
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
        else:
            log.warning("track_id=%d: not ignored, not autoregressive, no bars selected — no-op", tp.id)

        # Attribute name + value range
        for k, v in tp.attributes.items():
            if attr_names and k not in attr_names:
                raise RequestValidationError(
                    f"track_id={tp.id}: unknown attribute '{k}' "
                    f"(known: {sorted(attr_names)})"
                )
            sz = attr_sizes.get(k)
            if sz is not None and not (0 <= int(v) < sz):
                raise RequestValidationError(
                    f"track_id={tp.id}: attribute '{k}'={v} out of range [0, {sz})"
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
