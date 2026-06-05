"""Tests for the training pipeline (trainer, data module, lightning module).

Marker strategy:
  inference — needs torch (and any GPU); skipped in cibuildwheel CI
  slow      — needs parquet data on disk; skipped everywhere without it

The plain (unmarked) tests cover unit-level concerns that have no external deps:
  - TrainConfig round-trip serialisation
  - _validate_train_config rejects incompatible encoder/config pairs
"""
from __future__ import annotations

import json
import math
import pathlib
import tempfile

import pytest
import torch

import midigpt._core as _core


# --------------------------------------------------------------------------- #
#  TrainConfig — unit tests, no parquet, no torch, no markers
# --------------------------------------------------------------------------- #

class TestTrainConfig:
    def test_defaults(self):
        from midigpt.training.trainer import TrainConfig

        cfg = TrainConfig()
        assert cfg.learning_rate == pytest.approx(5e-5)
        assert cfg.precision in ("fp16", "bf16", "fp32")
        assert cfg.num_epochs >= 1
        assert cfg.per_device_batch_size >= 1

    def test_from_json_file(self, tmp_path):
        from midigpt.training.trainer import TrainConfig

        data = {"learning_rate": 1e-4, "n_layer": 2, "n_embd": 64, "n_head": 2}
        p = tmp_path / "cfg.json"
        p.write_text(json.dumps(data))
        cfg = TrainConfig.from_file(str(p))
        assert cfg.learning_rate == pytest.approx(1e-4)
        assert cfg.n_layer == 2

    def test_unknown_keys_ignored(self, tmp_path):
        from midigpt.training.trainer import TrainConfig

        p = tmp_path / "cfg.json"
        p.write_text(json.dumps({"learning_rate": 3e-4, "not_a_field": 999}))
        cfg = TrainConfig.from_file(str(p))
        assert cfg.learning_rate == pytest.approx(3e-4)


class TestValidateTrainConfig:
    def test_infill_rejected_without_support(self, ghost_config_json):
        from midigpt.training.trainer import TrainConfig, _validate_train_config

        cfg = TrainConfig(infill_probability=0.5)
        enc = json.loads(ghost_config_json)
        enc["supports_infill"] = False
        with pytest.raises(ValueError, match="supports_infill"):
            _validate_train_config(cfg, enc)

    def test_no_infill_passes(self, ghost_config_json):
        from midigpt.training.trainer import TrainConfig, _validate_train_config

        cfg = TrainConfig(infill_probability=0.0, mask_apply_probability=0.0)
        _validate_train_config(cfg, json.loads(ghost_config_json))


# --------------------------------------------------------------------------- #
#  LightningModule — forward/backward, no parquet needed
# --------------------------------------------------------------------------- #

@pytest.mark.inference
class TestLightningModule:
    def test_training_step_loss_finite(self, tiny_gpt2, ghost_tokenizer):
        """One synthetic training step should produce a finite loss."""
        import dataclasses
        from midigpt.training.lightning_module import MidiGPTLightningModule

        @dataclasses.dataclass
        class _Cfg:
            learning_rate: float = 1e-3
            weight_decay: float = 0.01
            warmup_steps: int = 0
            lr_scheduler_type: str = "constant"

        lit = MidiGPTLightningModule(tiny_gpt2, _Cfg())
        lit.total_steps = 10
        lit.train()

        B, T = 2, 32
        V = ghost_tokenizer.vocab_size()
        ids = torch.randint(0, V, (B, T))
        labels = ids.clone()

        batch = {"input_ids": ids, "labels": labels}
        loss = lit.training_step(batch, 0)
        assert torch.isfinite(loss), f"loss is not finite: {loss}"
        assert loss > 0


# --------------------------------------------------------------------------- #
#  MidiGPTDataset + DataModule — need parquet on disk
# --------------------------------------------------------------------------- #

@pytest.mark.slow
@pytest.mark.inference
class TestMidiGPTDataset:
    def test_dataset_loads_and_filters(self, ghost_tokenizer, training_parquet):
        from midigpt.training.dataset import MidiGPTDataset

        ds = MidiGPTDataset(
            str(training_parquet),
            ghost_tokenizer,
            infill_probability=0.0,
            mask_bar_config=None,
            max_seq_len=128,
            max_tracks=4,
            min_tracks=1,
            min_fill_ratio=0.5,
        )
        assert len(ds) > 0, "Dataset is empty after filtering"

    def test_dataset_item_shape(self, ghost_tokenizer, training_parquet):
        from midigpt.training.dataset import MidiGPTDataset

        ds = MidiGPTDataset(
            str(training_parquet),
            ghost_tokenizer,
            infill_probability=0.0,
            mask_bar_config=None,
            max_seq_len=64,
            max_tracks=4,
            min_tracks=1,
            min_fill_ratio=0.5,
        )
        # Find a non-None sample.
        sample = None
        for i in range(min(50, len(ds))):
            s = ds[i]
            if s is not None:
                sample = s
                break
        assert sample is not None, "No valid sample in first 50 rows"
        assert "input_ids" in sample
        assert len(sample["input_ids"]) <= 64

    def test_data_module_setup(self, ghost_tokenizer, training_parquet):
        from midigpt.training.data_module import MidiGPTDataModule

        dm = MidiGPTDataModule(
            train_path=str(training_parquet),
            tokenizer=ghost_tokenizer,
            infill_probability=0.0,
            mask_bar_config=None,
            max_seq_len=64,
            max_tracks=4,
            min_tracks=1,
            min_fill_ratio=0.5,
            per_device_batch_size=2,
            num_workers=0,
            pin_memory=False,
        )
        dm.setup()
        assert dm.train_dataset_size > 0
        dl = dm.train_dataloader()
        batch = next(iter(dl))
        assert "input_ids" in batch


@pytest.mark.slow
@pytest.mark.inference
def test_training_smoke(ghost_tokenizer, training_parquet, tmp_path):
    """Full 1-step Lightning training loop — verifies end-to-end integration."""
    import dataclasses
    import math

    import lightning as L

    from midigpt.inference.model.gpt2 import GPT2Config, GPT2LMHeadModel
    from midigpt.training.data_module import MidiGPTDataModule
    from midigpt.training.lightning_module import MidiGPTLightningModule

    gpt2_cfg = GPT2Config(
        vocab_size=ghost_tokenizer.vocab_size(),
        n_positions=64,
        n_embd=32,
        n_layer=2,
        n_head=2,
    )
    model = GPT2LMHeadModel(gpt2_cfg)

    @dataclasses.dataclass
    class _Cfg:
        learning_rate: float = 1e-3
        weight_decay: float = 0.01
        warmup_steps: int = 0
        lr_scheduler_type: str = "constant"

    dm = MidiGPTDataModule(
        train_path=str(training_parquet),
        tokenizer=ghost_tokenizer,
        infill_probability=0.0,
        mask_bar_config=None,
        max_seq_len=64,
        max_tracks=4,
        min_tracks=1,
        min_fill_ratio=0.5,
        per_device_batch_size=2,
        num_workers=0,
        pin_memory=False,
    )
    dm.setup()

    lit = MidiGPTLightningModule(model, _Cfg())
    lit.total_steps = 2

    (tmp_path / "checkpoints").mkdir(parents=True, exist_ok=True)
    trainer = L.Trainer(
        max_steps=2,
        precision="32",
        log_every_n_steps=1,
        default_root_dir=str(tmp_path),
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        limit_val_batches=0,
        num_sanity_val_steps=0,
    )
    trainer.fit(lit, dm)
    # Verify the model saved to a bundle.
    out = tmp_path / "smoke.safetensors"
    import json as _json
    enc = _json.loads(ghost_tokenizer._vocab.config().to_json())
    model.save_pretrained(str(out), encoder_config=enc)
    assert out.exists() and out.stat().st_size > 0
