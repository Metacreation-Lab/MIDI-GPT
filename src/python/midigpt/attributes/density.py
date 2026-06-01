from midigpt._types import Score
from midigpt.attributes.base import BaseAttribute


class NoteDensity(BaseAttribute):
    name = "note_density"
    token_type = "NoteDensity"
    level = "track"
    track_type = "melodic"
    size = 10

    def compute(self, score: Score, track_idx: int, bar_idx: int | None = None) -> float | int:
        track = score.tracks[track_idx]
        notes = sum(len(bar.notes) for bar in track.bars)
        bars = len(track.bars)
        return notes / bars if bars > 0 else 0

    def quantize(self, value: float | int) -> int:
        return min(int(value), 127)
