import random
from midigpt_refactor._types import Score
from midigpt_refactor.augmentation.base import BaseTransform

class TrackPermutation(BaseTransform):
    def __call__(self, score: Score) -> Score:
        random.shuffle(score.tracks)
        return score
