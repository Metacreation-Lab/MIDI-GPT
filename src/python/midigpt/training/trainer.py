from __future__ import annotations
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

log = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    # ── Output ───────────────────────────────────────────────────────────
    output_dir: str = "checkpoints"

    # ── Encoder ──────────────────────────────────────────────────────────
    encoder_config_path: str = ""

    # ── Optimiser ────────────────────────────────────────────────────────
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0          # gradient clipping (0 = disabled)
    warmup_steps: int = 500
    lr_scheduler_type: str = "linear"   # "linear" | "cosine" | "constant"

    # ── Training loop ────────────────────────────────────────────────────
    num_epochs: int = 10
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    seed: int = 42
    num_workers: int = 4

    # ── Precision ────────────────────────────────────────────────────────
    precision: Literal["fp16", "bf16", "fp32"] = "fp16"

    # ── Sequence length ───────────────────────────────────────────────────
    max_seq_len: int = 2048

    # ── Checkpointing / logging ──────────────────────────────────────────
    save_steps: int = 1000
    eval_steps: int = 500
    logging_steps: int = 50
    logger: Literal["tensorboard", "wandb", "none"] = "none"

    # ── Window / track sampling ───────────────────────────────────────────
    max_tracks: int = 12
    min_tracks: int = 1
    min_fill_ratio: float = 0.75

    # ── Infill vs. autoregressive sampling ───────────────────────────────
    infill_probability: float = 0.75

    # ── MaskBar config (applied to infill samples) ───────────────────────
    mask_apply_probability: float = 0.5
    mask_type: int = 2
    mask_bar_fraction: float = 0.25
    mask_max_lookahead: int = 4


def _precision_str(precision: str) -> str:
    return {"fp16": "16-mixed", "bf16": "bf16-mixed", "fp32": "32"}[precision]


def _build_logger(logger_type: str, output_dir: str):
    if logger_type == "tensorboard":
        from lightning.pytorch.loggers import TensorBoardLogger
        return TensorBoardLogger(save_dir=output_dir, name="logs")
    if logger_type == "wandb":
        from lightning.pytorch.loggers import WandbLogger
        return WandbLogger(save_dir=output_dir)
    return False   # Lightning disables logging when logger=False


def train(config: TrainConfig, train_path: str, eval_path: str | None = None):
    """Train GPT2LMHeadModel using PyTorch Lightning."""
    try:
        import lightning as L
        from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
    except ImportError:
        raise ImportError("pip install midigpt[train]")

    import midigpt._core as _core
    from midigpt.tokenizer.tokenizer import Tokenizer
    from midigpt.augmentation.mask_bar import MaskBarConfig
    from midigpt.inference.model import GPT2Config, GPT2LMHeadModel
    from midigpt.training.lightning_module import MidiGPTLightningModule
    from midigpt.training.data_module import MidiGPTDataModule

    L.seed_everything(config.seed, workers=True)

    encoder_config = _core.EncoderConfig.from_json(
        Path(config.encoder_config_path).read_text()
    )
    tokenizer = Tokenizer(encoder_config)

    mask_cfg = MaskBarConfig(
        apply_probability=config.mask_apply_probability,
        mode=config.mask_type,
        bar_fraction=config.mask_bar_fraction,
        max_lookahead=config.mask_max_lookahead,
    ) if config.infill_probability > 0.0 else None

    data_module = MidiGPTDataModule(
        train_path=train_path,
        tokenizer=tokenizer,
        mask_bar_config=mask_cfg,
        max_seq_len=config.max_seq_len,
        max_tracks=config.max_tracks,
        min_tracks=config.min_tracks,
        min_fill_ratio=config.min_fill_ratio,
        infill_probability=config.infill_probability,
        per_device_batch_size=config.per_device_batch_size,
        num_workers=config.num_workers,
        eval_path=eval_path,
    )
    # Setup datasets so we can compute total_steps for the LR scheduler.
    data_module.setup()
    steps_per_epoch = math.ceil(
        data_module.train_dataset_size
        / config.per_device_batch_size
        / config.gradient_accumulation_steps
    )
    total_steps = steps_per_epoch * config.num_epochs

    gpt2_cfg = GPT2Config(
        vocab_size=tokenizer.vocab_size(),
        n_positions=config.max_seq_len,
    )
    model = GPT2LMHeadModel(gpt2_cfg)
    model.encoder_config = encoder_config
    log.info("Model params: %d", sum(p.numel() for p in model.parameters()))

    lit_module = MidiGPTLightningModule(model, config)
    lit_module.total_steps = total_steps

    callbacks = [
        ModelCheckpoint(
            dirpath=Path(config.output_dir) / "checkpoints",
            every_n_train_steps=config.save_steps,
            save_top_k=-1,           # keep all checkpoints
            filename="step={step}",
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    trainer = L.Trainer(
        max_epochs=config.num_epochs,
        precision=_precision_str(config.precision),
        accumulate_grad_batches=config.gradient_accumulation_steps,
        gradient_clip_val=config.max_grad_norm if config.max_grad_norm > 0 else None,
        val_check_interval=config.eval_steps if eval_path else None,
        log_every_n_steps=config.logging_steps,
        default_root_dir=config.output_dir,
        callbacks=callbacks,
        logger=_build_logger(config.logger, config.output_dir),
    )

    trainer.fit(lit_module, data_module)

    # Save final packed inference bundle.
    import json
    enc_cfg = model.encoder_config
    if hasattr(enc_cfg, "to_json"):
        enc_cfg = json.loads(enc_cfg.to_json())
    final_path = Path(config.output_dir) / "model_final.pt"
    model.save_pretrained(str(final_path), encoder_config=enc_cfg)
    log.info("Training complete. Final bundle: %s", final_path)
