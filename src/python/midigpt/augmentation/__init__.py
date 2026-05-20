from midigpt.augmentation.base import BaseTransform, AugmentationPipeline
from midigpt.augmentation.transpose import Transpose
from midigpt.augmentation.velocity import VelocityScale
from midigpt.augmentation.track_permutation import TrackPermutation
from midigpt.augmentation.score_window import select_window
from midigpt.augmentation.instrument_swap import InstrumentSwap
from midigpt.augmentation.mask_bar import MaskBar, MaskBarConfig

__all__ = [
    "BaseTransform",
    "AugmentationPipeline",
    "Transpose",
    "VelocityScale",
    "TrackPermutation",
    "select_window",
    "InstrumentSwap",
    "MaskBar",
    "MaskBarConfig",
]
