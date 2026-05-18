import random
from midigpt_refactor._types import Score
from midigpt_refactor.augmentation.base import BaseTransform

class VelocityScale(BaseTransform):
    def __init__(self, factor: float | tuple[float, float]):
        self.factor = factor

    def __call__(self, score: Score) -> Score:
        if isinstance(self.factor, float):
            scale = self.factor
        else:
            scale = random.uniform(self.factor[0], self.factor[1])
            
        for track in score.tracks:
            for bar in track.bars:
                for note in bar.notes:
                    note.velocity = max(1, min(127, int(note.velocity * scale)))
        return score
