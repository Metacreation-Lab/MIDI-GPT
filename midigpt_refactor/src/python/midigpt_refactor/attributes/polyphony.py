from typing import Optional
from midigpt_refactor._types import Score
from midigpt_refactor.attributes.base import BaseAttribute

class OnsetPolyphony(BaseAttribute):
    name       = "onset_polyphony"
    token_type = "OnsetPolyphony"
    level      = "track"
    track_type = "melodic"
    size       = 10

    def compute(self, score: Score, track_idx: int, bar_idx: Optional[int] = None) -> float | int:
        track = score.tracks[track_idx]
        max_poly = 0
        for bar in track.bars:
            onsets = {}
            for note in bar.notes:
                onsets[note.onset_ticks] = onsets.get(note.onset_ticks, 0) + 1
            if onsets:
                max_poly = max(max_poly, max(onsets.values()))
        return max_poly

    def quantize(self, value: float | int) -> int:
        return min(int(value), 127)
