import random

from midigpt._types import Score
from midigpt.augmentation.base import BaseTransform


class InstrumentSwap(BaseTransform):
    def __init__(self, mapping: dict[int, list[int]]):
        self.mapping = mapping

    def __call__(self, score: Score) -> Score:
        for track in score.tracks:
            if track.instrument in self.mapping:
                track.instrument = random.choice(self.mapping[track.instrument])
        return score
