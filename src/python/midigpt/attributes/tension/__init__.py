import os
import tempfile

import numpy as np
import symusic

from midigpt._types import Score
from midigpt.attributes.base import BaseAttribute
from midigpt.attributes.tension._features import compute_track_tension
from midigpt.attributes.tension._timing import (
    _load_symusic_maps,
    _ticks_to_seconds,
    get_bar_starts_seconds,
    interval_level_average,
)

_TENSION_CACHE: dict = {}

# Interior decile edges of N(0, 1): scipy.stats.norm.ppf([0.1, 0.2, …, 0.9]).
# The tension model z-score normalizes its output per track (see _model.py),
# so each track's values are ~N(0, 1). Equi-probable (decile) bins use the
# token budget far better than equi-distant ones, which crowd the middle bins.
_TENSION_BIN_EDGES = np.array([
    -1.2815515655446004,
    -0.8416212335729143,
    -0.5244005127080407,
    -0.2533471031357997,
    0.0,
    0.2533471031357997,
    0.5244005127080407,
    0.8416212335729143,
    1.2815515655446004,
])


def score_to_temp_midi(score: Score) -> str:
    s = symusic.Score(score.resolution)
    for track in score.tracks:
        is_drum_track = track.track_type == "drum"
        t = symusic.Track(is_drum=is_drum_track)
        for bar in track.bars:
            for n in bar.notes:
                t.notes.append(
                    symusic.Note(
                        time=n.onset_ticks,
                        duration=n.duration_ticks,
                        pitch=n.pitch,
                        velocity=n.velocity,
                    )
                )
        s.tracks.append(t)

    s.tempos.append(symusic.Tempo(time=0, qpm=120))
    abs_tick = 0
    if score.tracks and score.tracks[0].bars:
        for bar in score.tracks[0].bars:
            bar_len = int(bar.ts_numerator * (score.resolution * 4 / bar.ts_denominator))
            s.time_signatures.append(
                symusic.TimeSignature(
                    time=abs_tick,
                    numerator=bar.ts_numerator,
                    denominator=bar.ts_denominator,
                )
            )
            abs_tick += bar_len

    s.sort()
    fd, path = tempfile.mkstemp(suffix=".mid")
    os.close(fd)
    s.dump_midi(path)
    return path


class Tension(BaseAttribute):
    name = "tension"
    token_type = "Tension"
    level = "bar"
    track_type = "melodic"
    size = 10

    def compute(self, score: Score, track_idx: int, bar_idx: int | None = None) -> float | int:
        if bar_idx is None:
            return 0.0

        cache_key = (id(score), track_idx)
        if cache_key not in _TENSION_CACHE:
            midi_path = score_to_temp_midi(score)
            track = score.tracks[track_idx]
            try:
                is_drum = track.track_type == "drum"
                try:
                    t_sec, tension = compute_track_tension(midi_path, track_idx, is_drum)
                except Exception:
                    # Farbood model can fail on extremely sparse tracks (numpy
                    # polyfit length mismatch on short feature vectors).
                    _TENSION_CACHE[cache_key] = []
                    return 0.0
                if t_sec.size > 0 and tension.size > 0:
                    bar_starts_sec = get_bar_starts_seconds(midi_path)
                    if bar_starts_sec.size > 0:
                        max_t = float(t_sec[-1])
                        bar_starts_sec = bar_starts_sec[bar_starts_sec <= max_t + 1e-9]
                        if bar_starts_sec.size >= 2:
                            _, bar_tension = interval_level_average(tension, t_sec, bar_starts_sec)
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
        return int(np.digitize(value, _TENSION_BIN_EDGES))


class TensionDrum(Tension):
    """Drum-track variant — same computation, distinct token type."""

    name = "tension_drum"
    token_type = "TensionDrum"
    track_type = "drum"
