"""LightningDataModule wrapping MidiGPTDataset + MidiGPTCollator."""

from __future__ import annotations

from typing import TYPE_CHECKING

import lightning as L
from torch.utils.data import DataLoader

from midigpt.training.collator import MidiGPTCollator
from midigpt.training.dataset import MidiGPTDataset

if TYPE_CHECKING:
    from midigpt.augmentation.mask_bar import MaskBarConfig
    from midigpt.tokenizer.tokenizer import Tokenizer


class MidiGPTDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_path: str | list[str],
        tokenizer: Tokenizer,
        # Infill training
        infill_probability: float = 0.75,
        infill_bar_fraction: float = 0.5,
        # Bar masking (independent of infill)
        mask_bar_config: MaskBarConfig | None = None,
        # Sequence budget
        max_seq_len: int = 2048,
        # Window / track sampling
        max_tracks: int = 12,
        min_tracks: int = 1,
        min_fill_ratio: float = 0.75,
        # DataLoader params
        per_device_batch_size: int = 4,
        num_workers: int = 4,
        pin_memory: bool = True,
        shuffle: bool = True,
        # Optional eval
        eval_path: str | None = None,
    ):
        super().__init__()
        self._train_path = train_path
        self._eval_path = eval_path
        self._tokenizer = tokenizer
        self._infill_prob = infill_probability
        self._infill_frac = infill_bar_fraction
        self._mask_cfg = mask_bar_config
        self._max_seq_len = max_seq_len
        self._max_tracks = max_tracks
        self._min_tracks = min_tracks
        self._fill_ratio = min_fill_ratio
        self._batch_size = per_device_batch_size
        self._num_workers = num_workers
        self._pin_memory = pin_memory
        self._shuffle = shuffle
        self._collator = MidiGPTCollator(max_seq_len=max_seq_len)
        self._train_ds: MidiGPTDataset | None = None
        self._eval_ds: MidiGPTDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        if self._train_ds is None:
            self._train_ds = MidiGPTDataset(
                self._train_path,
                self._tokenizer,
                infill_probability=self._infill_prob,
                infill_bar_fraction=self._infill_frac,
                mask_bar_config=self._mask_cfg,
                max_seq_len=self._max_seq_len,
                max_tracks=self._max_tracks,
                min_tracks=self._min_tracks,
                min_fill_ratio=self._fill_ratio,
            )
        if self._eval_path and self._eval_ds is None:
            self._eval_ds = MidiGPTDataset(
                self._eval_path,
                self._tokenizer,
                infill_probability=0.0,
                mask_bar_config=None,
                max_seq_len=self._max_seq_len,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train_ds,
            batch_size=self._batch_size,
            shuffle=self._shuffle,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            collate_fn=self._collator,
            persistent_workers=self._num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader | None:
        if self._eval_ds is None:
            return None
        return DataLoader(
            self._eval_ds,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            collate_fn=self._collator,
            persistent_workers=self._num_workers > 0,
        )

    @property
    def train_dataset_size(self) -> int:
        """Number of training samples (available after setup())."""
        return len(self._train_ds) if self._train_ds else 0
