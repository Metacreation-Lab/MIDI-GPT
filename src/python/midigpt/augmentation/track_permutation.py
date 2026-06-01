import random

from midigpt._types import Score
from midigpt.augmentation.base import BaseTransform


class TrackPermutation(BaseTransform):
    def __call__(self, score: Score) -> Score:
        random.shuffle(score.tracks)
        return score
