import logging
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    output_dir: str = "checkpoints"
    learning_rate: float = 5e-5
    num_epochs: int = 10
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    warmup_steps: int = 500
    weight_decay: float = 0.01
    max_seq_len: int = 2048
    save_steps: int = 1000
    eval_steps: int = 500
    logging_steps: int = 50
    fp16: bool = True
    seed: int = 42
    encoder_config_path: str = ""
    mask_augmentation: bool = False
    mask_apply_probability: float = 0.5
    mask_type: int = 2
    mask_bar_fraction: float = 0.25
    mask_max_lookahead: int = 4


def train(config: TrainConfig, train_path: str, eval_path: str | None = None):
    """Launch training using HuggingFace Trainer."""
    try:
        import torch
        from transformers import (
            GPT2Config, GPT2LMHeadModel, Trainer, TrainingArguments
        )
    except ImportError:
        raise ImportError("pip install midigpt[train]")

    from midigpt_refactor.tokenizer.tokenizer import Tokenizer
    from midigpt_refactor.training.dataset import MidiGPTDataset
    from midigpt_refactor.training.collator import MidiGPTCollator
    from midigpt_refactor.augmentation.base import AugmentationPipeline
    from midigpt_refactor.augmentation.mask_bar import MaskBar, MaskBarConfig
    import midigpt_refactor._core as _core

    encoder_config = _core.EncoderConfig.from_json(
        Path(config.encoder_config_path).read_text()
    )
    tokenizer = Tokenizer(encoder_config)

    augmenter = AugmentationPipeline.default_training()
    if config.mask_augmentation:
        augmenter._transforms.append(MaskBar(MaskBarConfig(
            apply_probability=config.mask_apply_probability,
            mode=config.mask_type,
            bar_fraction=config.mask_bar_fraction,
            max_lookahead=config.mask_max_lookahead,
        )))

    train_dataset = MidiGPTDataset(
        train_path, tokenizer, augmenter, config.max_seq_len
    )
    eval_dataset = (
        MidiGPTDataset(eval_path, tokenizer, None, config.max_seq_len)
        if eval_path else None
    )

    model_config = GPT2Config(
        vocab_size=tokenizer.vocab_size(),
        n_positions=config.max_seq_len,
        bos_token_id=0,
        eos_token_id=tokenizer.vocab_size() - 1,
    )
    model = GPT2LMHeadModel(model_config)
    log.info("Model params: %d", sum(p.numel() for p in model.parameters()))

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.per_device_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        weight_decay=config.weight_decay,
        fp16=config.fp16,
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
