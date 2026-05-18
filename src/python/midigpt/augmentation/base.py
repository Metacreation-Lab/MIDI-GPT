from abc import ABC, abstractmethod
from midigpt._types import Score
import copy

class BaseTransform(ABC):
    @abstractmethod
    def __call__(self, score: Score) -> Score: ...

class AugmentationPipeline:
    def __init__(self, transforms: list[BaseTransform]):
        self._transforms = transforms

    def __call__(self, score: Score) -> Score:
        score = copy.deepcopy(score)
        for t in self._transforms:
            score = t(score)
        return score

    @staticmethod
    def default_training() -> "AugmentationPipeline":
        from midigpt.augmentation import (
            Transpose, VelocityScale, TrackPermutation, BarWindow
        )
        return AugmentationPipeline([
            Transpose(range(-6, 7)),     # ±6 semitones, drums excluded
            VelocityScale((0.8, 1.2)),   # ±20%
            TrackPermutation(),          # shuffle track order
            BarWindow(num_bars=16),      # random 16-bar window
        ])
