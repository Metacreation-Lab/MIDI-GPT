"""
midigpt.realtime_state

PieceState — central musical state for the MIDI-GPT OSC server.

Owns all tracks, bars, and events.  All public methods are thread-safe.
Serializes to flat-pool format for sample_multi_step and deserializes
generated results back to inline-event format.

Design decisions (see docs/realtime_technical_plan.md §4):
- Events are stored *inline* per bar as dicts {pitch, velocity, time, internal_duration}.
- Flat-pool serialization (piece['events'] + integer indices per bar) is built on-demand
  in to_piece_dict(), just before json.dumps().  No global index state to maintain.
- time/duration conversion: onset/duration normalized floats → integer ticks via
  ts_numerator, ts_denominator, and resolution (Option B decision).
"""

import threading
from typing import Dict, List, Optional, Tuple

def bar_ticks(ts_num: int, ts_den: int, resolution: int) -> int:
    """Ticks per bar: ts_num/ts_den of a whole note = ts_num * 4 * res / ts_den."""
    return int(ts_num * 4 * resolution / ts_den)


# ---------------------------------------------------------------------------
# Per-track parameter defaults
# ---------------------------------------------------------------------------

_AGENT_PARAM_DEFAULTS: Dict = {
    "temperature": 1.0,
    "polyphony_hard_limit": 0,
    "min_polyphony_q": 0,
    "max_polyphony_q": 0,
    "min_note_duration_q": 0,
    "max_note_duration_q": 0,
    "min_pitch": 0,
    "max_pitch": 127,
    "key_signature": 0,
    "onset_density": 0,
    "onset_polyphony_min": 0,
    "onset_polyphony_max": 0,
    "density": 0,  # drum track only
}

_COND_PARAM_DEFAULTS: Dict = {
    "ignore": 0,
    "temperature": 1.0,
}


# ---------------------------------------------------------------------------
# TrackInfo
# ---------------------------------------------------------------------------

class TrackInfo:
    """Holds all state for one track."""

    __slots__ = ("track_id", "instrument", "track_type", "is_agent",
                 "piece_idx", "bars", "params")

    def __init__(self, track_id: int, instrument: int, track_type: int,
                 is_agent: bool, piece_idx: int) -> None:
        self.track_id = track_id
        self.instrument = instrument    # MIDI program (int)
        self.track_type = track_type    # 10=melodic, 11=drum
        self.is_agent = is_agent
        self.piece_idx = piece_idx      # index in piece['tracks']
        # bars[b] = {'ts_numerator': int, 'ts_denominator': int,
        #            'events': [{'pitch':…,'velocity':…,'time':…,'internal_duration':…}]}
        self.bars: List[dict] = []
        self.params: Dict = dict(
            _AGENT_PARAM_DEFAULTS if is_agent else _COND_PARAM_DEFAULTS
        )


# ---------------------------------------------------------------------------
# PieceState
# ---------------------------------------------------------------------------

class PieceState:
    """
    Central musical state.  Thread-safe.

    Lifecycle:
      create_track() × N  →  push_note() / end_bar() repeatedly  →
      extend_for_generation() + to_piece_dict() + build_status()  →
      merge_generated()
    """

    def __init__(self, resolution: int = 12) -> None:
        self.resolution = resolution
        self._lock = threading.Lock()

        self._tracks: Dict[int, TrackInfo] = {}     # track_id → TrackInfo
        self._agent_track_id: Optional[int] = None

        # Pending notes not yet committed by /bar/end:
        # (track_id, bar_index) → [event dict, …]
        self._pending: Dict[Tuple[int, int], List[dict]] = {}

        # Per-bar time signatures committed by /bar/end:
        # bar_index → (ts_num, ts_den)
        self._ts: Dict[int, Tuple[int, int]] = {}

        # Number of bars fully finalized (max bar_index seen + 1)
        self.bars_completed: int = 0

    # ── Read-only accessors ───────────────────────────────────────────────────

    @property
    def agent_track_id(self) -> Optional[int]:
        return self._agent_track_id

    @property
    def num_tracks(self) -> int:
        return len(self._tracks)

    def has_agent(self) -> bool:
        return self._agent_track_id is not None

    def has_conditioning_tracks(self) -> bool:
        return any(not t.is_agent for t in self._tracks.values())

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _sorted_tracks(self) -> List[TrackInfo]:
        return sorted(self._tracks.values(), key=lambda t: t.piece_idx)

    def _last_ts(self) -> Tuple[int, int]:
        """Most recently seen time signature, or 4/4 if none."""
        if self._ts:
            return self._ts[max(self._ts)]
        return (4, 4)

    def _get_ts(self, bar_index: int) -> Tuple[int, int]:
        return self._ts.get(bar_index, self._last_ts())

    def _extend_all_to_locked(self, num_bars: int) -> None:
        """Pad all tracks to num_bars. Must be called with _lock held."""
        ts_n, ts_d = self._last_ts()
        for info in self._tracks.values():
            while len(info.bars) < num_bars:
                b = len(info.bars)
                bn, bd = self._ts.get(b, (ts_n, ts_d))
                info.bars.append(
                    {"ts_numerator": bn, "ts_denominator": bd, "events": []}
                )

    # ── Track management ──────────────────────────────────────────────────────

    def create_track(self, track_id: int, instrument: int, track_type: int,
                     is_agent: bool) -> Optional[str]:
        """Add a track.  Returns None on success, error string on failure."""
        with self._lock:
            if track_id in self._tracks:
                return f"Duplicate track ID: {track_id}"
            if is_agent and self._agent_track_id is not None:
                return "Only one agent track allowed per session"
            piece_idx = len(self._tracks)
            info = TrackInfo(track_id, instrument, track_type, is_agent, piece_idx)
            # Back-fill empty bars for already-completed bars
            self._tracks[track_id] = info   # temporarily add so _extend sees it
            self._extend_all_to_locked(self.bars_completed)
            if is_agent:
                self._agent_track_id = track_id
        return None

    def remove_track(self, track_id: int) -> Optional[str]:
        """Remove a conditioning track.  Returns None on success."""
        with self._lock:
            if track_id not in self._tracks:
                return f"Unknown track ID: {track_id}"
            info = self._tracks[track_id]
            if info.is_agent:
                return "Cannot remove agent track during session"
            removed_idx = info.piece_idx
            del self._tracks[track_id]
            # Compact piece indices
            for ti in self._tracks.values():
                if ti.piece_idx > removed_idx:
                    ti.piece_idx -= 1
            # Drop pending notes for removed track
            keys_to_drop = [k for k in self._pending if k[0] == track_id]
            for k in keys_to_drop:
                del self._pending[k]
        return None

    def set_track_param(self, track_id: int, name: str, value) -> Optional[str]:
        """Update a per-track parameter.  Returns None on success."""
        with self._lock:
            if track_id not in self._tracks:
                return f"Unknown track ID: {track_id}"
            ti = self._tracks[track_id]
            if name not in ti.params:
                return f"Unknown per-track parameter: {name!r}"
            ti.params[name] = value
        return None

    def reset_track_param(self, track_id: int, name: str) -> Optional[str]:
        """Reset a per-track parameter to its default."""
        with self._lock:
            if track_id not in self._tracks:
                return f"Unknown track ID: {track_id}"
            ti = self._tracks[track_id]
            defaults = _AGENT_PARAM_DEFAULTS if ti.is_agent else _COND_PARAM_DEFAULTS
            if name not in defaults:
                return f"Unknown per-track parameter: {name!r}"
            ti.params[name] = defaults[name]
        return None

    # ── Note accumulation ─────────────────────────────────────────────────────

    def push_note(self, track_id: int, pitch: int, velocity: int,
                  onset: float, duration: float, bar_index: int) -> Optional[str]:
        """
        Accumulate one note for (track_id, bar_index).

        onset and duration are normalized [0.0, 1.0).  Tick conversion is
        intentionally deferred to end_bar so that non-4/4 time signatures are
        handled correctly: notes often arrive before /bar/end registers the
        bar's ts, and converting early using the last-known ts (which defaults
        to 4/4) produces out-of-range ticks for shorter bars (e.g., 3/4, 6/8).

        Returns None on success, "agent_track_note_ignored" for the agent track,
        or an error string.
        """
        with self._lock:
            if track_id not in self._tracks:
                return f"Unknown track ID: {track_id}"
            info = self._tracks[track_id]
            if info.is_agent:
                return "agent_track_note_ignored"
            # Store normalized values — converted to ticks in end_bar once ts is known.
            event = {
                "pitch":             int(pitch),
                "velocity":          int(velocity),
                "_onset_norm":       float(onset),
                "_duration_norm":    float(duration),
            }
            key = (track_id, bar_index)
            if key not in self._pending:
                self._pending[key] = []
            self._pending[key].append(event)
        return None

    # ── Bar finalization ──────────────────────────────────────────────────────

    def end_bar(self, bar_index: int, ts_num: int, ts_den: int) -> None:
        """
        Finalize bar bar_index: store time sig, flush pending notes, extend
        all tracks.  Called when /bar/end arrives.

        Tick conversion happens here so the correct ts is always used,
        regardless of note arrival order relative to /bar/end.
        """
        with self._lock:
            self._ts[bar_index] = (ts_num, ts_den)
            ticks = bar_ticks(ts_num, ts_den, self.resolution)
            self._extend_all_to_locked(bar_index + 1)

            for track_id, info in self._tracks.items():
                # Apply time sig to this bar (may have been filled with default earlier)
                info.bars[bar_index]["ts_numerator"] = ts_num
                info.bars[bar_index]["ts_denominator"] = ts_den
                # Flush pending notes, converting normalized onset/duration to ticks now.
                key = (track_id, bar_index)
                if key in self._pending:
                    converted = []
                    for ev in self._pending.pop(key):
                        onset_norm    = ev.pop("_onset_norm")
                        duration_norm = ev.pop("_duration_norm")
                        ev["time"]              = int(onset_norm * ticks)
                        ev["internal_duration"] = max(1, int(duration_norm * ticks))
                        converted.append(ev)
                    info.bars[bar_index]["events"] = converted

            self.bars_completed = max(self.bars_completed, bar_index + 1)

    # ── Pre-inference extension ───────────────────────────────────────────────

    def extend_for_generation(self, target_bar: int, num_anticipation: int) -> None:
        """
        Ensure all tracks have at least target_bar + num_anticipation bars.
        Empty bars beyond bars_completed use the last known time signature.
        """
        needed = target_bar + num_anticipation
        with self._lock:
            self._extend_all_to_locked(needed)

    # ── Serialization ─────────────────────────────────────────────────────────

    def to_score_and_request(self, playhead: int, target_bar: Optional[int], num_anticipation: int, mask_gap: bool, params: dict):
        from midigpt_refactor._types import Score, Track, Bar, Note
        from midigpt_refactor.inference.config import GenerationRequest, TrackPrompt, SamplingConfig

        with self._lock:
            model_dim = params.get("model_dim", 4)
            # Compute sliding context window: only pass the last model_dim bars
            # to the engine so the context length stays bounded as the piece grows.
            end_bar = (target_bar + num_anticipation) if target_bar is not None else 0
            window_start = max(0, end_bar - model_dim)

            score = Score(resolution=self.resolution, tracks=[])
            tracks_params = []

            for info in self._sorted_tracks():
                track = Track(
                    track_type="drum" if info.track_type == 11 else "melodic",
                    instrument=info.instrument,
                    bars=[],
                )
                for b_idx, bar_info in enumerate(info.bars[window_start:]):
                    actual_b = b_idx + window_start
                    future = False
                    if info.is_agent:
                        if target_bar is not None and actual_b >= target_bar + num_anticipation:
                            future = True
                        elif mask_gap and playhead <= actual_b < (target_bar or 0):
                            future = True
                    else:
                        if actual_b >= playhead:
                            future = True

                    bar = Bar(
                        ts_numerator=bar_info.get("ts_numerator", 4),
                        ts_denominator=bar_info.get("ts_denominator", 4),
                        future=future,
                        notes=[],
                    )
                    for ev in bar_info.get("events", []):
                        bar.notes.append(Note(
                            pitch=ev["pitch"],
                            velocity=ev["velocity"],
                            onset_ticks=ev["time"],
                            duration_ticks=ev.get("internal_duration", 1),
                            delta=0,
                        ))
                    track.bars.append(bar)
                score.tracks.append(track)

                # Build TrackPrompt with per-track attribute params
                attrs: Dict = {}
                if info.is_agent:
                    selected_bars = list(range(
                        target_bar - window_start,
                        min(target_bar + num_anticipation - window_start, len(info.bars) - window_start)
                    )) if target_bar is not None else []

                    p = info.params
                    if p.get("polyphony_hard_limit", 0) != 0:
                        attrs["onset_polyphony"] = p["polyphony_hard_limit"]
                    if p.get("min_polyphony_q", 0) != 0:
                        attrs["min_polyphony"] = p["min_polyphony_q"]
                    if p.get("max_polyphony_q", 0) != 0:
                        attrs["max_polyphony"] = p["max_polyphony_q"]
                    if p.get("min_note_duration_q", 0) != 0:
                        attrs["min_note_duration"] = p["min_note_duration_q"]
                    if p.get("max_note_duration_q", 0) != 0:
                        attrs["max_note_duration"] = p["max_note_duration_q"]
                    if p.get("onset_density", 0) != 0:
                        attrs["note_density"] = p["onset_density"]
                    if p.get("key_signature", 0) != 0:
                        attrs["key_signature"] = p["key_signature"]

                    tracks_params.append(TrackPrompt(
                        id=info.piece_idx,
                        autoregressive=True,
                        ignore=False,
                        bars=selected_bars,
                        attributes=attrs,
                    ))
                else:
                    tracks_params.append(TrackPrompt(
                        id=info.piece_idx,
                        autoregressive=False,
                        ignore=bool(info.params.get("ignore", 0)),
                        bars=[],
                    ))

            cfg = SamplingConfig(
                temperature=params.get("temperature", 1.0),
                seed=params.get("sampling_seed", -1),
                model_dim=params.get("model_dim", 4),
                max_attempts=1,        # realtime: accept first attempt, move on
                silence_check=False,   # realtime: don't block on silent bars
                novelty_check=False,
            )
            return score, GenerationRequest(tracks=tracks_params, config=cfg), window_start

    # ── Result merge ──────────────────────────────────────────────────────────

    def merge_generated(
        self, res_score, target_bar: int, num_anticipation: int, window_start: int = 0
    ) -> List[Tuple[int, List[dict], Tuple[int, int]]]:
        """
        Write back generated bars from the result piece into the agent track.
        window_start — bar offset used when building the windowed score; result
                       track bars are indexed from 0 = window_start in global space.
        """
        result: List[Tuple[int, List[dict], Tuple[int, int]]] = []
        with self._lock:
            if self._agent_track_id is None:
                return result
            agent = self._tracks[self._agent_track_id]
            res_track = res_score.tracks[agent.piece_idx]

            for b_off in range(num_anticipation):
                b_global = target_bar + b_off
                b_local  = b_global - window_start
                if b_local < 0 or b_local >= len(res_track.bars):
                    break
                # Extend agent track if needed (should already be long enough)
                while len(agent.bars) <= b_global:
                    ts_n, ts_d = self._last_ts()
                    agent.bars.append(
                        {"ts_numerator": ts_n, "ts_denominator": ts_d, "events": []}
                    )
                res_bar = res_track.bars[b_local]
                inline = []
                for n in res_bar.notes:
                    inline.append({
                        "pitch": n.pitch,
                        "velocity": n.velocity,
                        "time": n.onset_ticks,
                        "internal_duration": n.duration_ticks
                    })
                agent.bars[b_global]["events"] = inline
                ts = self._ts.get(b_global, self._last_ts())
                result.append((b_global, inline, ts))

        return result
