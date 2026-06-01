import copy
from abc import ABC, abstractmethod

from midigpt._types import Score


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
        """Pitch and velocity augmentations applied to the full score before windowing.

        Window selection, track sampling, and infill masking are handled by
        MidiGPTDataset, which needs to retry on overflow — those steps cannot
        be inside a stateless transform pipeline.
        """
        from midigpt.augmentation import Transpose, VelocityScale

        return AugmentationPipeline(
            [
                Transpose(range(-6, 7)),  # ±6 semitones, drums excluded
                VelocityScale((0.8, 1.2)),  # ±20%
            ]
        )
