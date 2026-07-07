from midigpt._types import Score
from midigpt.attributes.base import BaseAttribute


class PitchClassSet(BaseAttribute):
    size = 13

    def __init__(self, level: str = "bar", track_type: str = "melodic"):
        self.level = level
        self.track_type = track_type
        self.name = "pitch_class_set" if level == "bar" else "pitch_class_set_track"
        self.token_type = "BarLevelPitchClassSet" if level == "bar" else "PitchClassSetTrack"

    def compute(self, score: Score, track_idx: int, bar_idx: int | None = None) -> float | int:
        track = score.tracks[track_idx]
        if self.level == "bar":
            if bar_idx is None or bar_idx >= len(track.bars):
                return 0
            pitches = {note.pitch % 12 for note in track.bars[bar_idx].notes}
        else:
            pitches = set()
            for bar in track.bars:
                for note in bar.notes:
                    pitches.add(note.pitch % 12)
        return len(pitches)

    def quantize(self, value: float | int) -> int:
        return min(int(value), 12)


# Backward-compatible alias
BarLevelPitchClassSet = PitchClassSet
