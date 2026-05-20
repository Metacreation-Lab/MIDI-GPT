import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

log = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    # ── Output ───────────────────────────────────────────────────────────
    output_dir: str = "checkpoints"

    # ── Encoder ──────────────────────────────────────────────────────────
    encoder_config_path: str = ""

    # ── Optimizer ────────────────────────────────────────────────────────
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0          # gradient clipping (0 = disabled)
    warmup_steps: int = 500
    lr_scheduler_type: str = "linear"   # "linear" | "cosine" | "constant" | etc.

    # ── Training loop ────────────────────────────────────────────────────
    num_epochs: int = 10
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    seed: int = 42

    # ── Precision ────────────────────────────────────────────────────────
    # "fp16" | "bf16" | "fp32"
    precision: Literal["fp16", "bf16", "fp32"] = "fp16"

    # ── Sequence length ───────────────────────────────────────────────────
    max_seq_len: int = 2048

    # ── Checkpointing / logging ──────────────────────────────────────────
    save_steps: int = 1000
    eval_steps: int = 500
    logging_steps: int = 50

    # ── Window / track sampling ───────────────────────────────────────────
    max_tracks: int = 12
    min_tracks: int = 1
    # Minimum fraction of bars that must be non-empty for a track to be
    # included in a window (0.75 = legacy default).
    min_fill_ratio: float = 0.75

    # ── Infill vs. autoregressive sampling ───────────────────────────────
    # Fraction of samples trained with infill (FillIn tokens).
    # 0.0 = always autoregressive; 0.75 = legacy default.
    infill_probability: float = 0.75

    # ── MaskBar config (applied to infill samples) ───────────────────────
    mask_apply_probability: float = 0.5  # gate: skip masking entirely
    mask_type: int = 2                   # 0=random 1=structured 2=mixed
    mask_bar_fraction: float = 0.25      # max fraction of bars to mask
    mask_max_lookahead: int = 4          # structured-mode lookahead bars


def train(config: TrainConfig, train_path: str, eval_path: str | None = None):
    """Launch training using HuggingFace Trainer."""
    try:
        import torch
        from transformers import Trainer, TrainingArguments
    except ImportError:
        raise ImportError("pip install midigpt[train]")

    import midigpt._core as _core
    from midigpt.tokenizer.tokenizer import Tokenizer
    from midigpt.training.dataset import MidiGPTDataset
    from midigpt.training.collator import MidiGPTCollator
    from midigpt.augmentation.mask_bar import MaskBarConfig
    from midigpt.inference.model import GPT2Config, GPT2LMHeadModel

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

    dataset_kwargs = dict(
        mask_bar_config=mask_cfg,
        max_seq_len=config.max_seq_len,
        max_tracks=config.max_tracks,
        min_tracks=config.min_tracks,
        min_fill_ratio=config.min_fill_ratio,
        infill_probability=config.infill_probability,
    )
    train_dataset = MidiGPTDataset(train_path, tokenizer, **dataset_kwargs)
    eval_dataset = (
        MidiGPTDataset(
            eval_path, tokenizer,
            mask_bar_config=None,
            max_seq_len=config.max_seq_len,
            infill_probability=0.0,
        )
        if eval_path else None
    )

    gpt2_cfg = GPT2Config(
        vocab_size=tokenizer.vocab_size(),
        n_positions=config.max_seq_len,
    )
    model = GPT2LMHeadModel(gpt2_cfg)
    model.encoder_config = encoder_config
    log.info("Model params: %d", sum(p.numel() for p in model.parameters()))

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.per_device_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        lr_scheduler_type=config.lr_scheduler_type,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm if config.max_grad_norm > 0 else None,
        fp16=(config.precision == "fp16"),
        bf16=(config.precision == "bf16"),
        save_steps=config.save_steps,
        eval_steps=config.eval_steps if eval_dataset else None,
        eval_strategy="steps" if eval_dataset else "no",
        logging_steps=config.logging_steps,
        seed=config.seed,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=MidiGPTCollator(max_seq_len=config.max_seq_len),
    )
    trainer.train()
    trainer.save_model()
    log.info("Training complete. Model saved to %s", config.output_dir)
