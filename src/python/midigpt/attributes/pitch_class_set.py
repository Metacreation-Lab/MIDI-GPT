from typing import Optional
from midigpt._types import Score
from midigpt.attributes.base import BaseAttribute

class BarLevelPitchClassSet(BaseAttribute):
    name       = "pitch_class_set"
    token_type = "PitchClassSet"
    level      = "bar"
    track_type = "melodic"
    size       = 13

    def compute(self, score: Score, track_idx: int, bar_idx: Optional[int] = None) -> float | int:
        if bar_idx is None:
            return 0
        track = score.tracks[track_idx]
        if bar_idx >= len(track.bars):
            return 0
        pitches = {note.pitch % 12 for note in track.bars[bar_idx].notes}
        return len(pitches)

    def quantize(self, value: float | int) -> int:
        return min(int(value), 127)
