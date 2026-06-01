import sys
import os
import tempfile
from typing import Optional
from midigpt._types import Score
from midigpt.attributes.base import BaseAttribute
import numpy as np
import symusic

# tension_model lives in a sibling repo. Import lazily so this module loads
# (and the Tension class can be registered) even when tension_model is not
# installed — only `.compute()` actually requires it.
_TENSION_MODEL_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "../../../../../tension_model"))

def _ttm():
    if _TENSION_MODEL_PATH not in sys.path:
        sys.path.append(_TENSION_MODEL_PATH)
    import testTensionModel as ttm
    return ttm

# Cache to avoid re-evaluating the whole track for every bar
_TENSION_CACHE = {}

def score_to_temp_midi(score: Score) -> str:
    s = symusic.Score(score.resolution)
    for track in score.tracks:
        is_drum_track = (track.track_type == "drum")
        t = symusic.Track(is_drum=is_drum_track)
        for bar in track.bars:
            for n in bar.notes:
                t.notes.append(symusic.Note(
                    time=n.onset_ticks,
                    duration=n.duration_ticks,
                    pitch=n.pitch,
                    velocity=n.velocity
                ))
        s.tracks.append(t)
    
    s.tempos.append(symusic.Tempo(time=0, qpm=120))
    abs_tick = 0
    if score.tracks and score.tracks[0].bars:
        for bar in score.tracks[0].bars:
            bar_len = int(bar.ts_numerator * (score.resolution * 4 / bar.ts_denominator))
            s.time_signatures.append(symusic.TimeSignature(
                time=abs_tick,
                numerator=bar.ts_numerator,
                denominator=bar.ts_denominator
            ))
            abs_tick += bar_len
            
    s.sort()
    fd, path = tempfile.mkstemp(suffix=".mid")
    os.close(fd)
    s.dump_midi(path)
    return path

def get_bar_starts_seconds(midi_path: str):
    tpq, tempo_ticks, mspq, ts_ticks, nums, dens, end_tick = _ttm()._load_symusic_maps(midi_path)
    bar_ticks = []
    for i in range(len(ts_ticks)):
        start = int(ts_ticks[i])
        end = int(ts_ticks[i + 1]) if i + 1 < len(ts_ticks) else end_tick
        num = int(nums[i])
        den = int(dens[i])
        ticks_per_bar = int(round(tpq * (4.0 / den) * num))
        if ticks_per_bar <= 0:
            continue
        t = start
        while t < end:
            bar_ticks.append(t)
            t += ticks_per_bar
    bar_ticks = np.unique(np.asarray(bar_ticks, dtype=np.int64))
    return _ttm()._ticks_to_seconds(bar_ticks, tempo_ticks, mspq, tpq)

class Tension(BaseAttribute):
    name       = "tension"
    token_type = "Tension"
    level      = "bar"
    track_type = "melodic"
    size       = 10

    def compute(self, score: Score, track_idx: int, bar_idx: Optional[int] = None) -> float | int:
        if bar_idx is None:
            return 0.0

        cache_key = (id(score), track_idx)
        if cache_key not in _TENSION_CACHE:
            midi_path = score_to_temp_midi(score)
            track = score.tracks[track_idx]
            try:
                is_drum_track = (track.track_type == "drum")
                t_sec, tension = _ttm().compute_track_tension(midi_path, track_idx, is_drum_track)
                if t_sec.size > 0 and tension.size > 0:
                    bar_starts_sec = get_bar_starts_seconds(midi_path)
                    if bar_starts_sec.size > 0:
                        max_t = float(t_sec[-1])
                        bar_starts_sec = bar_starts_sec[bar_starts_sec <= max_t + 1e-9]
                        if bar_starts_sec.size >= 2:
                            _, bar_tension = _ttm().interval_level_average(tension, t_sec, bar_starts_sec)
                            _TENSION_CACHE[cache_key] = bar_tension.tolist()
                        else:
                            _TENSION_CACHE[cache_key] = []
                    else:
                        _TENSION_CACHE[cache_key] = []
                else:
                    _TENSION_CACHE[cache_key] = []
            finally:
                os.remove(midi_path)
                
        bar_tensions = _TENSION_CACHE.get(cache_key, [])
        if bar_idx < len(bar_tensions):
            return bar_tensions[bar_idx]
        return 0.0

    def quantize(self, value: float | int) -> int:
        # Original model normalized features but kept tension raw.
        # Assuming tension roughly spans [-2, 2], map to [0, 9]
        norm_val = (value + 2.0) / 4.0
        return max(0, min(9, int(norm_val * 10)))


class TensionDrum(Tension):
    """Drum-track variant of Tension. Same computation; distinct token
    type so the model can specialize attention separately for drum tracks."""
    name       = "tension_drum"
    token_type = "TensionDrum"
    track_type = "drum"
