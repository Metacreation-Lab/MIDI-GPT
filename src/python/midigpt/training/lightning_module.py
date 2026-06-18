"""LightningModule wrapping GPT2LMHeadModel for supervised language-model training."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING

import lightning as L
import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from midigpt.inference.model.gpt2 import GPT2LMHeadModel
    from midigpt.training.trainer import TrainConfig


def _make_lr_lambda(warmup_steps: int, total_steps: int, scheduler_type: str):
    """Return a LambdaLR multiplier function for common scheduler types."""

    def linear(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 1.0 - progress)

    def cosine(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    def constant(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return 1.0

    return {"linear": linear, "cosine": cosine, "constant": constant}.get(scheduler_type, linear)


class MidiGPTLightningModule(L.LightningModule):
    def __init__(self, model: GPT2LMHeadModel, config: TrainConfig):
        super().__init__()
        self.model = model
        self.config = config
        # total_steps is set by MidiGPTDataModule.setup() before fit() starts
        self.total_steps: int = 0

    # ------------------------------------------------------------------ #
    #  Forward / loss
    # ------------------------------------------------------------------ #
    def _step(self, batch: dict, stage: str) -> torch.Tensor:
        input_ids = batch["input_ids"]
        labels = batch["labels"]

        logits, _ = self.model(input_ids)
        # Shift: predict token i+1 from token i
        shift_logits = logits[:, :-1].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        self.log(
            f"{stage}/loss",
            loss,
            on_step=(stage == "train"),
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        self._step(batch, "val")

    # ------------------------------------------------------------------ #
    #  Optimiser + scheduler
    # ------------------------------------------------------------------ #
    def configure_optimizers(self):
        cfg = self.config
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        # total_steps is computed from the full (un-sharded) dataset. Under DDP
        # the dataset is sharded across ranks, so the real number of optimizer
        # steps per epoch is divided by world_size; the cosine/linear schedule
        # length must match or the LR decays too slowly on multi-GPU.
        world_size = max(1, getattr(self.trainer, "world_size", 1))
        total_steps = math.ceil(self.total_steps / world_size)
        lr_lambda = _make_lr_lambda(cfg.warmup_steps, total_steps, cfg.lr_scheduler_type)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    # ------------------------------------------------------------------ #
    #  Save packed inference bundle alongside every Lightning checkpoint
    # ------------------------------------------------------------------ #
    def on_save_checkpoint(self, checkpoint: dict) -> None:
        """Write a safetensors inference bundle next to the .ckpt file."""
        ckpt_dir = self.trainer.checkpoint_callback.dirpath
        if ckpt_dir is None:
            return
        step = self.trainer.global_step
        bundle_path = Path(ckpt_dir) / f"model-step={step}.safetensors"
        bundle_path.parent.mkdir(parents=True, exist_ok=True)
        enc_cfg = self.model.encoder_config
        if isinstance(enc_cfg, object) and hasattr(enc_cfg, "to_json"):
            enc_cfg = json.loads(enc_cfg.to_json())
        self.model.save_pretrained(str(bundle_path), encoder_config=enc_cfg)
