from midigpt.training.trainer import TrainConfig, train

# collator and dataset require 'pyarrow' and 'datasets' (from [train] extra).
try:
    from midigpt.training.collator import MidiGPTCollator
    from midigpt.training.dataset import _SCORE_SCHEMA, DatasetBuilder, MidiGPTDataset
    _has_train_deps = True
except ImportError:
    MidiGPTCollator = None  # type: ignore[assignment,misc]
    _SCORE_SCHEMA = None  # type: ignore[assignment,misc]
    DatasetBuilder = None  # type: ignore[assignment,misc]
    MidiGPTDataset = None  # type: ignore[assignment,misc]
    _has_train_deps = False

# data_module and lightning_module require 'lightning'.
try:
    from midigpt.training.data_module import MidiGPTDataModule
    from midigpt.training.lightning_module import MidiGPTLightningModule
    _has_lightning = True
except ImportError:
    MidiGPTDataModule = None  # type: ignore[assignment,misc]
    MidiGPTLightningModule = None  # type: ignore[assignment,misc]
    _has_lightning = False

__all__ = [
    "TrainConfig",
    "train",
]

if _has_train_deps:
    __all__.extend([
        "_SCORE_SCHEMA",
        "DatasetBuilder",
        "MidiGPTCollator",
        "MidiGPTDataset",
    ])

if _has_lightning:
    __all__.extend([
        "MidiGPTDataModule",
        "MidiGPTLightningModule",
    ])
