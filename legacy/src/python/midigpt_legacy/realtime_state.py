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

# ---------------------------------------------------------------------------
# GM instrument name table
# Index = MIDI program number (0–127).  Values are GM_TYPE enum string names
# used in StatusTrack.instrument.
# ---------------------------------------------------------------------------
_GM_INST_NAMES: List[str] = [
    "acoustic_grand_piano", "bright_acoustic_piano", "electric_grand_piano",
    "honky_tonk_piano", "electric_piano_1", "electric_piano_2", "harpsichord",
    "clavi", "celesta", "glockenspiel", "music_box", "vibraphone", "marimba",
    "xylophone", "tubular_bells", "dulcimer", "drawbar_organ", "percussive_organ",
    "rock_organ", "church_organ", "reed_organ", "accordion", "harmonica",
    "tango_accordion", "acoustic_guitar_nylon", "acoustic_guitar_steel",
    "electric_guitar_jazz", "electric_guitar_clean", "electric_guitar_muted",
    "overdriven_guitar", "distortion_guitar", "guitar_harmonics", "acoustic_bass",
    "electric_bass_finger", "electric_bass_pick", "fretless_bass", "slap_bass_1",
    "slap_bass_2", "synth_bass_1", "synth_bass_2", "violin", "viola", "cello",
    "contrabass", "tremolo_strings", "pizzicato_strings", "orchestral_harp",
    "timpani", "string_ensemble_1", "string_ensemble_2", "synth_strings_1",
    "synth_strings_2", "choir_aahs", "voice_oohs", "synth_voice", "orchestra_hit",
    "trumpet", "trombone", "tuba", "muted_trumpet", "french_horn", "brass_section",
    "synth_brass_1", "synth_brass_2", "soprano_sax", "alto_sax", "tenor_sax",
    "baritone_sax", "oboe", "english_horn", "bassoon", "clarinet", "piccolo",
    "flute", "recorder", "pan_flute", "blown_bottle", "shakuhachi", "whistle",
    "ocarina", "lead_1_square", "lead_2_sawtooth", "lead_3_calliope",
    "lead_4_chiff", "lead_5_charang", "lead_6_voice", "lead_7_fifths",
    "lead_8_bass__lead", "pad_1_new_age", "pad_2_warm", "pad_3_polysynth",
    "pad_4_choir", "pad_5_bowed", "pad_6_metallic", "pad_7_halo", "pad_8_sweep",
    "fx_1_rain", "fx_2_soundtrack", "fx_3_crystal", "fx_4_atmosphere",
    "fx_5_brightness", "fx_6_goblins", "fx_7_echoes", "fx_8_sci_fi",
    "sitar", "banjo", "shamisen", "koto", "kalimba", "bag_pipe", "fiddle",
    "shanai", "tinkle_bell", "agogo", "steel_drums", "woodblock", "taiko_drum",
    "melodic_tom", "synth_drum", "reverse_cymbal", "guitar_fret_noise",
    "breath_noise", "seashore", "bird_tweet", "telephone_ring", "helicopter",
    "applause", "gunshot",
]


def instrument_gm_name(program: int) -> str:
    """MIDI program → GM_TYPE enum string for StatusTrack.instrument."""
    if 0 <= program < len(_GM_INST_NAMES):
        return _GM_INST_NAMES[program]
    return "any"


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

    def to_piece_dict(self) -> dict:
        """
        Snapshot the current state as a flat-pool piece dict.

        Events stored inline are packed into a flat events list; bar.events
        fields become integer index arrays.  Safe to call without holding the
        lock (takes _lock internally).
        """
        with self._lock:
            all_events: List[dict] = []
            tracks_out: List[dict] = []

            for info in self._sorted_tracks():
                bars_out = []
                for bar in info.bars:
                    event_indices = []
                    for ev in bar.get("events", []):
                        event_indices.append(len(all_events))
                        all_events.append({
                            "pitch": ev["pitch"],
                            "velocity": ev["velocity"],
                            "time": ev["time"],
                            "internal_duration": ev.get("internal_duration", 1),
                        })
                    bar_out: dict = {
                        "ts_numerator": bar.get("ts_numerator", 4),
                        "ts_denominator": bar.get("ts_denominator", 4),
                    }
                    if event_indices:
                        bar_out["events"] = event_indices
                    bars_out.append(bar_out)

                tracks_out.append({
                    "track_type": info.track_type,
                    "instrument": info.instrument,
                    "bars": bars_out,
                })

            return {
                "tracks": tracks_out,
                "events": all_events,
                "resolution": self.resolution,
            }

    def build_status(self, playhead: int, target_bar: Optional[int],
                     num_anticipation: int, mask_gap: bool) -> dict:
        """
        Build the status dict for sample_multi_step.

        Every field of midi.StatusTrack / midi.StatusBar that has a meaningful
        default is set explicitly — relying on protobuf defaults has bitten us
        before (e.g. polyphony_hard_limit=0 → zero notes generated).

        playhead       — first unfinished bar (= bars_completed at trigger time)
        target_bar     — first bar to generate (None = no generation this step)
        num_anticipation — number of bars to generate (j)
        mask_gap       — hide agent gap bars [playhead, target_bar) with future=True
        """

        def _track_type_enum(t: int) -> str:
            return "STANDARD_DRUM_TRACK" if t == 11 else "STANDARD_TRACK"

        def _bar_defaults(ts_n: int, ts_d: int, future: bool,
                          drums: bool = False) -> dict:
            # Reasonable mid-range controls. Using *_ANY causes the C++ side to
            # recompute these from the (possibly empty) bar content, which
            # collapses to zero and pushes the model toward emptiness.
            if drums:
                onset_density = "BAR_LEVEL_ONSET_DENSITY_EIGHT"
                onset_poly_min = "BAR_LEVEL_ONSET_POLYPHONY_TWO"
                onset_poly_max = "BAR_LEVEL_ONSET_POLYPHONY_FOUR"
            else:
                onset_density = "BAR_LEVEL_ONSET_DENSITY_FOUR"
                onset_poly_min = "BAR_LEVEL_ONSET_POLYPHONY_ONE"
                onset_poly_max = "BAR_LEVEL_ONSET_POLYPHONY_TWO"
            return {
                "ts_numerator":        ts_n,
                "ts_denominator":      ts_d,
                "future":              future,
                "onset_density":       onset_density,
                "onset_polyphony_min": onset_poly_min,
                "onset_polyphony_max": onset_poly_max,
                "tension":             "DECILE_LEVEL_ANY",
                "tension_drum":        "DECILE_LEVEL_ANY",
                "pitch_class_set":     [False] * 12,
            }

        def _track_defaults(track_id: int, track_type: int, instrument_gm: int,
                            selected_bars: list, suffix_ar: bool, ignore: bool,
                            params: dict, status_bars: list) -> dict:
            drums = (track_type == 11)
            # Reasonable mid-range track-level controls. *_ANY triggers C++
            # recomputation from the (possibly empty) track, which collapses to
            # zero and pushes the model toward generating nothing.
            if drums:
                density            = "DENSITY_SIX"
                min_poly           = "POLYPHONY_ONE"
                max_poly           = "POLYPHONY_FOUR"
                min_dur            = "DURATION_THIRTY_SECOND"
                max_dur            = "DURATION_SIXTEENTH"
                onset_poly_min     = "BAR_LEVEL_ONSET_POLYPHONY_TWO"
                onset_poly_max     = "BAR_LEVEL_ONSET_POLYPHONY_FOUR"
                onset_density      = "BAR_LEVEL_ONSET_DENSITY_EIGHT"
                pitch_class_count  = "PITCH_CLASS_COUNT_FOUR"
            else:
                density            = "DENSITY_FIVE"
                min_poly           = "POLYPHONY_ONE"
                max_poly           = "POLYPHONY_TWO"
                min_dur            = "DURATION_SIXTEENTH"
                max_dur            = "DURATION_QUARTER"
                onset_poly_min     = "BAR_LEVEL_ONSET_POLYPHONY_ONE"
                onset_poly_max     = "BAR_LEVEL_ONSET_POLYPHONY_TWO"
                onset_density      = "BAR_LEVEL_ONSET_DENSITY_FOUR"
                pitch_class_count  = "PITCH_CLASS_COUNT_SEVEN"
            return {
                # identity
                "track_id":              track_id,
                "track_type":            _track_type_enum(track_type),
                "instrument":            instrument_gm_name(instrument_gm),
                # selection / generation mode
                "selected_bars":         selected_bars,
                "autoregressive":        False,
                "suffix_autoregressive": suffix_ar,
                "ignore":                ignore,
                # hard sampling constraints
                "polyphony_hard_limit":  int(params.get("polyphony_hard_limit") or 10),
                "temperature":           float(params.get("temperature", 1.0)),
                # attribute controls — explicit values (not ANY) so the C++
                # side doesn't recompute them from empty content.
                "density":               density,
                "min_polyphony_q":       min_poly,
                "max_polyphony_q":       max_poly,
                "min_note_duration_q":   min_dur,
                "max_note_duration_q":   max_dur,
                "onset_polyphony_min":   onset_poly_min,
                "onset_polyphony_max":   onset_poly_max,
                "onset_density":         onset_density,
                "min_pitch":             int(params.get("min_pitch", 0)),
                "max_pitch":             int(params.get("max_pitch", 127)),
                "key_signature":         "KEY_SIGNATURE_ANY",
                "note_density_level":    density,
                "pitch_class_count":     pitch_class_count,
                # per-bar status
                "bars": status_bars,
            }

        with self._lock:
            total_bars = max(
                (len(t.bars) for t in self._tracks.values()), default=0
            )
            status_tracks = []

            for info in self._sorted_tracks():
                tb = total_bars
                ts_n, ts_d = self._last_ts()

                drums = (info.track_type == 11)
                if info.is_agent:
                    sel = [False] * tb
                    status_bars: List[dict] = []

                    if target_bar is not None:
                        for b in range(target_bar, min(target_bar + num_anticipation, tb)):
                            sel[b] = True
                        for b in range(tb):
                            future = (
                                b >= target_bar + num_anticipation
                                or (mask_gap and playhead <= b < target_bar)
                            )
                            bts = self._ts.get(b, (ts_n, ts_d))
                            status_bars.append(_bar_defaults(bts[0], bts[1], future, drums))
                    else:
                        for b in range(tb):
                            bts = self._ts.get(b, (ts_n, ts_d))
                            status_bars.append(_bar_defaults(bts[0], bts[1], False, drums))

                    st = _track_defaults(
                        track_id=info.piece_idx,
                        track_type=info.track_type,
                        instrument_gm=info.instrument,
                        selected_bars=sel,
                        suffix_ar=True,
                        ignore=False,
                        params=info.params,
                        status_bars=status_bars,
                    )
                    _apply_agent_params(st, info.params, info.track_type)

                else:
                    status_bars = []
                    for b in range(tb):
                        bts = self._ts.get(b, (ts_n, ts_d))
                        status_bars.append(_bar_defaults(bts[0], bts[1], b >= playhead, drums))
                    st = _track_defaults(
                        track_id=info.piece_idx,
                        track_type=info.track_type,
                        instrument_gm=info.instrument,
                        selected_bars=[False] * tb,
                        suffix_ar=False,
                        ignore=bool(info.params.get("ignore", 0)),
                        params=info.params,
                        status_bars=status_bars,
                    )
                    if info.params.get("temperature", 1.0) != 1.0:
                        st["temperature"] = info.params["temperature"]

                status_tracks.append(st)

            return {"tracks": status_tracks}

    # ── Result merge ──────────────────────────────────────────────────────────

    def merge_generated(
        self, res_piece: dict, target_bar: int, num_anticipation: int
    ) -> List[Tuple[int, List[dict], Tuple[int, int]]]:
        """
        Write back generated bars from the result piece into the agent track.

        Returns a list of (bar_index, inline_events, (ts_num, ts_den)) for each
        generated bar — used by the caller to send /generated/note messages.
        Inline events are dicts {pitch, velocity, time, internal_duration}.
        """
        result: List[Tuple[int, List[dict], Tuple[int, int]]] = []
        with self._lock:
            if self._agent_track_id is None:
                return result
            agent = self._tracks[self._agent_track_id]
            res_events = res_piece.get("events", [])
            res_agent_bars = res_piece["tracks"][agent.piece_idx].get("bars", [])

            window_size = len(res_agent_bars)
            for b_off in range(num_anticipation):
                b_global = target_bar + b_off
                # C++ result is a model_dim-bar window; generated bars are at the end.
                res_idx = window_size - num_anticipation + b_off
                if res_idx < 0 or res_idx >= window_size:
                    break
                # Extend agent track if needed (should already be long enough)
                while len(agent.bars) <= b_global:
                    ts_n, ts_d = self._last_ts()
                    agent.bars.append(
                        {"ts_numerator": ts_n, "ts_denominator": ts_d, "events": []}
                    )
                res_bar = res_agent_bars[res_idx]
                inline = [
                    {k: v for k, v in res_events[i].items()
                     if k in ("pitch", "velocity", "time", "internal_duration")}
                    for i in res_bar.get("events", [])
                    if i < len(res_events)
                ]
                agent.bars[b_global]["events"] = inline
                ts = self._ts.get(b_global, self._last_ts())
                result.append((b_global, inline, ts))

        return result


# ---------------------------------------------------------------------------
# Helper: apply agent per-track params to status dict (skip default values)
# ---------------------------------------------------------------------------

def _apply_agent_params(st: dict, params: dict, track_type: int) -> None:
    p = params
    if p.get("temperature", 1.0) != 1.0:
        st["temperature"] = p["temperature"]
    if p.get("polyphony_hard_limit", 0) != 0:
        st["polyphony_hard_limit"] = p["polyphony_hard_limit"]
    if p.get("min_polyphony_q", 0) != 0:
        st["min_polyphony_q"] = p["min_polyphony_q"]
    if p.get("max_polyphony_q", 0) != 0:
        st["max_polyphony_q"] = p["max_polyphony_q"]
    if p.get("min_note_duration_q", 0) != 0:
        st["min_note_duration_q"] = p["min_note_duration_q"]
    if p.get("max_note_duration_q", 0) != 0:
        st["max_note_duration_q"] = p["max_note_duration_q"]
    if p.get("min_pitch", 0) != 0:
        st["min_pitch"] = p["min_pitch"]
    if p.get("max_pitch", 127) != 127:
        st["max_pitch"] = p["max_pitch"]
    if p.get("key_signature", 0) != 0:
        st["key_signature"] = p["key_signature"]
    if p.get("onset_density", 0) != 0:
        st["onset_density"] = p["onset_density"]
    if p.get("onset_polyphony_min", 0) != 0:
        st["onset_polyphony_min"] = p["onset_polyphony_min"]
    if p.get("onset_polyphony_max", 0) != 0:
        st["onset_polyphony_max"] = p["onset_polyphony_max"]
    if track_type == 11 and p.get("density", 0) != 0:
        st["density"] = p["density"]
