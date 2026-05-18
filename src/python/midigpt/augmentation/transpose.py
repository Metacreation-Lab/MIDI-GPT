import random
from midigpt._types import Score
from midigpt.augmentation.base import BaseTransform

class Transpose(BaseTransform):
    def __init__(self, semitones: int | range):
        self.semitones = semitones

    def __call__(self, score: Score) -> Score:
        if isinstance(self.semitones, int):
            shift = self.semitones
        else:
            shift = random.choice(list(self.semitones))
            
        for track in score.tracks:
            if track.track_type == "drum":
                continue
            for bar in track.bars:
                for note in bar.notes:
                    note.pitch = max(0, min(127, note.pitch + shift))
        return score
