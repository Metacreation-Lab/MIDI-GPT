import threading
from typing import Dict, List, Optional, Tuple

from midigpt._types import Score, Track, Bar, Note
from midigpt.inference.config import GenerationRequest, TrackPrompt, SamplingConfig

_AGENT_PARAM_DEFAULTS: Dict = {
    "temperature": 1.0,
    "min_pitch": 0,
    "max_pitch": 127,
}

_COND_PARAM_DEFAULTS: Dict = {
    "ignore": 0,
    "temperature": 1.0,
}


def bar_ticks(ts_num: int, ts_den: int, resolution: int) -> int:
    return int(ts_num * 4 * resolution / ts_den)


class TrackInfo:
    __slots__ = ("track_id", "instrument", "track_type", "is_agent",
                 "piece_idx", "bars", "params")

    def __init__(self, track_id: int, instrument: int, track_type: int,
                 is_agent: bool, piece_idx: int) -> None:
        self.track_id = track_id
        self.instrument = instrument
        self.track_type = track_type
        self.is_agent = is_agent
        self.piece_idx = piece_idx
        self.bars: List[dict] = []
        self.params: Dict = dict(
            _AGENT_PARAM_DEFAULTS if is_agent else _COND_PARAM_DEFAULTS
        )


class PieceState:
    def __init__(self, resolution: int = 12) -> None:
        self.resolution = resolution
        self._lock = threading.Lock()
        self._tracks: Dict[int, TrackInfo] = {}
        self._agent_track_id: Optional[int] = None
        self._pending: Dict[Tuple[int, int], List[dict]] = {}
        self._ts: Dict[int, Tuple[int, int]] = {}
        self.bars_completed: int = 0

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

    def _sorted_tracks(self) -> List[TrackInfo]:
        return sorted(self._tracks.values(), key=lambda t: t.piece_idx)

    def _last_ts(self) -> Tuple[int, int]:
        if self._ts:
            return self._ts[max(self._ts)]
        return (4, 4)

    def _get_ts(self, bar_index: int) -> Tuple[int, int]:
        return self._ts.get(bar_index, self._last_ts())

    def _extend_all_to_locked(self, num_bars: int) -> None:
        ts_n, ts_d = self._last_ts()
        for info in self._tracks.values():
            while len(info.bars) < num_bars:
                b = len(info.bars)
                bn, bd = self._ts.get(b, (ts_n, ts_d))
                info.bars.append(
                    {"ts_numerator": bn, "ts_denominator": bd, "events": []}
                )

    def create_track(self, track_id: int, instrument: int, track_type: int,
                     is_agent: bool) -> Optional[str]:
        with self._lock:
            if track_id in self._tracks:
                return f"Duplicate track ID: {track_id}"
            if is_agent and self._agent_track_id is not None:
                return "Only one agent track allowed per session"
            piece_idx = len(self._tracks)
            info = TrackInfo(track_id, instrument, track_type, is_agent, piece_idx)
            self._tracks[track_id] = info
            self._extend_all_to_locked(self.bars_completed)
            if is_agent:
                self._agent_track_id = track_id
        return None

    def remove_track(self, track_id: int) -> Optional[str]:
        with self._lock:
            if track_id not in self._tracks:
                return f"Unknown track ID: {track_id}"
            info = self._tracks[track_id]
            if info.is_agent:
                return "Cannot remove agent track during session"
            removed_idx = info.piece_idx
            del self._tracks[track_id]
            for ti in self._tracks.values():
                if ti.piece_idx > removed_idx:
                    ti.piece_idx -= 1
            keys_to_drop = [k for k in self._pending if k[0] == track_id]
            for k in keys_to_drop:
                del self._pending[k]
        return None

    def set_track_param(self, track_id: int, name: str, value) -> Optional[str]:
        with self._lock:
            if track_id not in self._tracks:
                return f"Unknown track ID: {track_id}"
            ti = self._tracks[track_id]
            if name not in ti.params:
                return f"Unknown per-track parameter: {name!r}"
            ti.params[name] = value
        return None

    def push_note(self, track_id: int, pitch: int, velocity: int,
                  onset: float, duration: float, bar_index: int) -> Optional[str]:
        with self._lock:
            if track_id not in self._tracks:
                return f"Unknown track ID: {track_id}"
            info = self._tracks[track_id]
            if info.is_agent:
                return "agent_track_note_ignored"
            event = {
                "pitch":          int(pitch),
                "velocity":       int(velocity),
                "_onset_norm":    float(onset),
                "_duration_norm": float(duration),
            }
            key = (track_id, bar_index)
            if key not in self._pending:
                self._pending[key] = []
            self._pending[key].append(event)
        return None

    def end_bar(self, bar_index: int, ts_num: int, ts_den: int) -> None:
        with self._lock:
            self._ts[bar_index] = (ts_num, ts_den)
            ticks = bar_ticks(ts_num, ts_den, self.resolution)
            self._extend_all_to_locked(bar_index + 1)

            for track_id, info in self._tracks.items():
                info.bars[bar_index]["ts_numerator"] = ts_num
                info.bars[bar_index]["ts_denominator"] = ts_den
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

    def extend_for_generation(self, target_bar: int, num_anticipation: int) -> None:
        needed = target_bar + num_anticipation
        with self._lock:
            self._extend_all_to_locked(needed)

    def to_score(self) -> Score:
        """Build a Score from current state (thread-safe snapshot)."""
        with self._lock:
            tracks_out: List[Track] = []
            for info in self._sorted_tracks():
                track_type = "drum" if info.track_type == 11 else "melodic"
                bars_out: List[Bar] = []
                for bar in info.bars:
                    ts_n = bar.get("ts_numerator", 4)
                    ts_d = bar.get("ts_denominator", 4)
                    notes: List[Note] = []
                    for ev in bar.get("events", []):
                        if ev.get("velocity", 0) > 0:
                            notes.append(Note(
                                pitch=ev["pitch"],
                                velocity=ev["velocity"],
                                onset_ticks=ev["time"],
                                duration_ticks=ev.get("internal_duration", 1),
                            ))
                    bars_out.append(Bar(
                        notes=notes,
                        ts_numerator=ts_n,
                        ts_denominator=ts_d,
                    ))
                tracks_out.append(Track(
                    bars=bars_out,
                    instrument=info.instrument,
                    track_type=track_type,
                ))
            return Score(tracks=tracks_out, resolution=self.resolution)

    def build_generation_request(
        self,
        target_bar: int,
        num_anticipation: int,
        params: dict,
    ) -> GenerationRequest:
        """Build GenerationRequest for the inference engine.

        Agent track: autoregressive=True, bars=list(range(target_bar, target_bar+num_anticipation))
        Conditioning tracks: bars=[], ignore=bool(params.get("ignore", 0))
        SamplingConfig: temperature, seed (from sampling_seed, -1=random), max_attempts,
                        bars_per_step=num_anticipation, tracks_per_step=1,
                        model_dim from params
        """
        with self._lock:
            track_prompts: List[TrackPrompt] = []
            for info in self._sorted_tracks():
                if info.is_agent:
                    track_prompts.append(TrackPrompt(
                        id=info.piece_idx,
                        bars=list(range(target_bar, target_bar + num_anticipation)),
                        autoregressive=True,
                        ignore=False,
                    ))
                else:
                    track_prompts.append(TrackPrompt(
                        id=info.piece_idx,
                        bars=[],
                        autoregressive=False,
                        ignore=bool(info.params.get("ignore", 0)),
                    ))

        raw_seed = int(params.get("sampling_seed", -1))
        config = SamplingConfig(
            temperature=float(params.get("temperature", 1.0)),
            seed=raw_seed,
            max_attempts=int(params.get("max_attempts", 3)),
            bars_per_step=num_anticipation,
            tracks_per_step=1,
            model_dim=int(params.get("model_dim", 4)),
        )
        return GenerationRequest(tracks=track_prompts, config=config)

    def merge_generated(
        self,
        result: Score,
        target_bar: int,
        num_anticipation: int,
        result_resolution: int,
    ) -> List[Tuple[int, List[dict], Tuple[int, int]]]:
        """Extract generated bars from result Score back into agent track state.

        Returns list of (bar_index, inline_events, (ts_num, ts_den)).
        inline_events = [{"pitch":, "velocity":, "time": onset_ticks, "internal_duration": duration_ticks}]
        result_resolution is the Score's resolution (for the caller to use in normalization).
        """
        output: List[Tuple[int, List[dict], Tuple[int, int]]] = []
        with self._lock:
            if self._agent_track_id is None:
                return output
            agent = self._tracks[self._agent_track_id]
            if agent.piece_idx >= len(result.tracks):
                return output
            res_track = result.tracks[agent.piece_idx]
            window_size = len(res_track.bars)

            for b_off in range(num_anticipation):
                b_global = target_bar + b_off
                res_idx = window_size - num_anticipation + b_off
                if res_idx < 0 or res_idx >= window_size:
                    break
                while len(agent.bars) <= b_global:
                    ts_n, ts_d = self._last_ts()
                    agent.bars.append(
                        {"ts_numerator": ts_n, "ts_denominator": ts_d, "events": []}
                    )
                res_bar = res_track.bars[res_idx]
                inline = [
                    {
                        "pitch": n.pitch,
                        "velocity": n.velocity,
                        "time": n.onset_ticks,
                        "internal_duration": n.duration_ticks,
                    }
                    for n in res_bar.notes
                    if n.velocity > 0
                ]
                agent.bars[b_global]["events"] = inline
                ts = self._ts.get(b_global, self._last_ts())
                output.append((b_global, inline, ts))

        return output
