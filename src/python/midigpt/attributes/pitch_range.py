from typing import Optional
from midigpt._types import Score
from midigpt.attributes.base import BaseAttribute

class PitchRange(BaseAttribute):
    name       = "pitch_range"
    token_type = "PitchRange"
    level      = "track"
    track_type = "melodic"
    size       = 128

    def compute(self, score: Score, track_idx: int, bar_idx: Optional[int] = None) -> float | int:
        track = score.tracks[track_idx]
        pitches = [note.pitch for bar in track.bars for note in bar.notes]
        if not pitches:
            return 0
        return max(pitches) - min(pitches)

    def quantize(self, value: float | int) -> int:
        return min(int(value), 127)
