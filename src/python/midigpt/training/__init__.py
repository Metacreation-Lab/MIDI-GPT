from midigpt.training.collator import MidiGPTCollator
from midigpt.training.dataset import _SCORE_SCHEMA, DatasetBuilder, MidiGPTDataset
from midigpt.training.trainer import TrainConfig, train

try:
    from midigpt.training.data_module import MidiGPTDataModule
    from midigpt.training.lightning_module import MidiGPTLightningModule
    _has_lightning = True
except ImportError:
    MidiGPTDataModule = None  # type: ignore[assignment,misc]
    MidiGPTLightningModule = None  # type: ignore[assignment,misc]
    _has_lightning = False

__all__ = [
    "_SCORE_SCHEMA",
    "DatasetBuilder",
    "MidiGPTCollator",
    "MidiGPTDataset",
    "TrainConfig",
    "train",
]

if _has_lightning:
    __all__.extend([
        "MidiGPTDataModule",
        "MidiGPTLightningModule",
    ])
