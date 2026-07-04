from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

log = logging.getLogger(__name__)

try:
    from lightning.pytorch.loggers import Logger as _LightningLogger
except ImportError:
    _LightningLogger = object  # type: ignore[assignment,misc]


class JSONLinesLogger(_LightningLogger):
    """Always-on file logger: appends one JSON object per step to metrics.jsonl."""

    def __init__(self, save_dir: str) -> None:
        super().__init__()
        self._path = Path(save_dir) / "metrics.jsonl"
        self._path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def name(self) -> str:
        return "jsonl"

    @property
    def version(self) -> str:
        return ""

    @property
    def save_dir(self) -> str:
        return str(self._path.parent)

    @property
    def root_dir(self) -> str:
        return str(self._path.parent)

    @property
    def log_dir(self) -> str:
        return str(self._path.parent)

    def log_hyperparams(self, params: Any, *args: Any, **kwargs: Any) -> None:
        pass

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        record = {"step": step, **metrics}
        with self._path.open("a") as f:
            f.write(json.dumps(record) + "\n")

    def save(self) -> None:
        pass

    def finalize(self, status: str) -> None:
        pass


@dataclass
class TrainConfig:
    # ── Output ───────────────────────────────────────────────────────────
    output_dir: str = "checkpoints"

    # ── Encoder ──────────────────────────────────────────────────────────
    encoder_config_path: str = ""

    # ── Optimiser ────────────────────────────────────────────────────────
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0  # gradient clipping (0 = disabled)
    warmup_steps: int = 500
    lr_scheduler_type: str = "linear"  # "linear" | "cosine" | "constant"

    # ── Training loop ────────────────────────────────────────────────────
    num_epochs: int = 10
    max_steps: int = 0  # if > 0, overrides num_epochs; LR schedule uses this as total_steps
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    seed: int = 42
    num_workers: int = 0  # use spawn context (data_module.py) to safely go above 0

    # ── Model architecture ────────────────────────────────────────────────
    n_embd: int = 512
    n_layer: int = 6
    n_head: int = 8

    # ── Precision ────────────────────────────────────────────────────────
    precision: Literal["fp16", "bf16", "fp32"] = "fp16"

    # ── Sequence length ───────────────────────────────────────────────────
    # Dataset-side cap on token sequences. Must not exceed the model's
    # max_context() (n_positions). Validated at training start.
    max_seq_len: int = 2048

    # ── Checkpointing / logging ──────────────────────────────────────────
    save_steps: int = 1000
    eval_steps: int = 500
    # Cap validation to this many batches per eval (0 = full validation set).
    # Validating the entire set every eval_steps is dominated by val cost when
    # the val set is large; a few hundred batches give a stable loss estimate.
    limit_val_batches: int = 0
    logging_steps: int = 50
    logger: Literal["tensorboard", "wandb", "none"] = "none"
    # WandB-specific (ignored when logger != "wandb")
    wandb_project: str = "midigpt"
    wandb_entity: str = ""  # defaults to your personal WandB account

    # ── Window / track sampling ───────────────────────────────────────────
    max_tracks: int = 12
    min_tracks: int = 1
    min_fill_ratio: float = 0.75

    # ── Infill training (independent of bar masking) ──────────────────────
    # Fraction of samples encoded with FillIn tokens. 0.0 = always AR.
    infill_probability: float = 0.75
    # Maximum per-cell infill density. Each sample draws p ~ Uniform(0, this),
    # then each (track, bar) cell is independently selected with probability p.
    infill_bar_fraction: float = 0.5

    # ── Bar masking (independent of infill) ───────────────────────────────
    # Fraction of samples where MASK_BAR is applied (gate inside MaskBarConfig).
    # Set mask_apply_probability=0.0 to disable masking entirely.
    mask_apply_probability: float = 0.5
    mask_mode: int = 2  # MaskMode: 0=RANDOM 1=STRUCTURED 2=MIXED
    mask_bar_fraction: float = 0.25
    mask_max_lookahead: int = 4

    @classmethod
    def from_file(cls, path: str) -> TrainConfig:
        """Load a TrainConfig from a JSON or YAML file. Unknown keys are ignored."""
        import json as _json

        p = Path(path)
        if p.suffix in (".yaml", ".yml"):
            try:
                import yaml

                data = yaml.safe_load(p.read_text())
            except ImportError:
                raise ImportError("pip install pyyaml to load YAML train configs") from None
        else:
            data = _json.loads(p.read_text())
        import dataclasses

        valid = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in valid})


def _validate_train_config(config: TrainConfig, encoder_config_json: dict) -> None:
    """Raise ValueError if TrainConfig requests features the encoder doesn't support."""
    if config.infill_probability > 0 and not encoder_config_json.get("supports_infill", False):
        raise ValueError(
            f"infill_probability={config.infill_probability} > 0 but the encoder config "
            f"has supports_infill=false. Set infill_probability=0.0 or use an "
            f"infill-capable checkpoint."
        )
    if config.mask_apply_probability > 0:
        # Build a minimal vocabulary to test for MaskBar token presence.
        import json as _json

        import midigpt._core as _core

        cfg = _core.EncoderConfig.from_json(_json.dumps(encoder_config_json))
        vocab = _core.Vocabulary(cfg)
        if not vocab.has(_core.TokenType.MaskBar):
            raise ValueError(
                f"mask_apply_probability={config.mask_apply_probability} > 0 but the "
                f"encoder vocab does not include the MaskBar token. "
                f"Set mask_apply_probability=0.0 or use a masking-capable checkpoint."
            )


def _precision_str(precision: str) -> str:
    return {"fp16": "16-mixed", "bf16": "bf16-mixed", "fp32": "32"}[precision]


def _build_loggers(config: TrainConfig) -> list:
    loggers: list = [JSONLinesLogger(config.output_dir)]

    if config.logger == "tensorboard":
        from lightning.pytorch.loggers import TensorBoardLogger

        loggers.append(TensorBoardLogger(save_dir=config.output_dir, name="logs"))
    elif config.logger == "wandb":
        import wandb as _wandb
        from lightning.pytorch.loggers import WandbLogger

        run_name = Path(config.output_dir).name
        init_kwargs: dict = dict(
            project=config.wandb_project,
            name=run_name,
            dir=config.output_dir,
        )
        if config.wandb_entity:
            init_kwargs["entity"] = config.wandb_entity
        try:
            run = _wandb.init(**init_kwargs)
        except Exception as exc:
            log.warning(
                "wandb online init failed (%s); switching to offline mode"
                " — run `wandb sync %s` after training to upload.",
                exc,
                config.output_dir,
            )
            run = _wandb.init(**init_kwargs, mode="offline")
        loggers.append(WandbLogger(experiment=run))

    return loggers


def _load_dotenv() -> None:
    """Load .env from the repo root if python-dotenv is available."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass


def train(
    config: TrainConfig,
    train_path: str,
    eval_path: str | None = None,
    resume_from: str | None = None,
):
    """Train GPT2LMHeadModel using PyTorch Lightning."""
    _load_dotenv()

    try:
        import lightning as L
        from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
    except ImportError:
        raise ImportError("pip install midigpt[train]") from None

    import midigpt._core as _core
    from midigpt.augmentation.mask_bar import MaskBarConfig, MaskMode
    from midigpt.inference.model import GPT2Config, GPT2LMHeadModel
    from midigpt.tokenizer.tokenizer import Tokenizer
    from midigpt.training.data_module import MidiGPTDataModule
    from midigpt.training.lightning_module import MidiGPTLightningModule

    L.seed_everything(config.seed, workers=True)

    import json as _json

    enc_path = Path(config.encoder_config_path)
    if enc_path.suffix == ".safetensors":
        from safetensors import safe_open

        with safe_open(str(enc_path), framework="pt") as _f:
            _meta = _f.metadata()
        enc_json_str = _meta.get("encoder_config", "{}")
    elif enc_path.suffix == ".pt":
        import torch as _torch

        _bundle = _torch.load(str(enc_path), map_location="cpu", weights_only=False)
        _enc = _bundle.get("encoder_config", {})
        enc_json_str = _json.dumps(_enc) if isinstance(_enc, dict) else _enc
    else:
        enc_json_str = enc_path.read_text()
    encoder_config = _core.EncoderConfig.from_json(enc_json_str)
    _validate_train_config(config, _json.loads(enc_json_str))
    tokenizer = Tokenizer(encoder_config)

    # Build the model first so we can validate max_seq_len.
    gpt2_cfg = GPT2Config(
        vocab_size=tokenizer.vocab_size(),
        n_positions=config.max_seq_len,
        n_embd=config.n_embd,
        n_layer=config.n_layer,
        n_head=config.n_head,
    )
    model = GPT2LMHeadModel(gpt2_cfg)
    model.encoder_config = encoder_config

    if config.max_seq_len > model.max_context():
        raise ValueError(
            f"max_seq_len={config.max_seq_len} exceeds the model's positional "
            f"budget ({model.max_context()}). Lower max_seq_len or increase "
            f"n_positions in the model config."
        )
    log.info("Model params: %d", sum(p.numel() for p in model.parameters()))

    mask_cfg = (
        MaskBarConfig(
            apply_probability=config.mask_apply_probability,
            mode=MaskMode(config.mask_mode),
            bar_fraction=config.mask_bar_fraction,
            max_lookahead=config.mask_max_lookahead,
        )
        if config.mask_apply_probability > 0.0
        else None
    )

    data_module = MidiGPTDataModule(
        train_path=train_path,
        tokenizer=tokenizer,
        infill_probability=config.infill_probability,
        infill_bar_fraction=config.infill_bar_fraction,
        mask_bar_config=mask_cfg,
        max_seq_len=config.max_seq_len,
        max_tracks=config.max_tracks,
        min_tracks=config.min_tracks,
        min_fill_ratio=config.min_fill_ratio,
        per_device_batch_size=config.per_device_batch_size,
        num_workers=config.num_workers,
        eval_path=eval_path,
    )
    data_module.setup()

    steps_per_epoch = math.ceil(
        data_module.train_dataset_size
        / config.per_device_batch_size
        / config.gradient_accumulation_steps
    )
    total_steps = config.max_steps if config.max_steps > 0 else steps_per_epoch * config.num_epochs

    lit_module = MidiGPTLightningModule(model, config)
    lit_module.total_steps = total_steps

    loggers = _build_loggers(config)
    callbacks = [
        ModelCheckpoint(
            dirpath=Path(config.output_dir) / "checkpoints",
            every_n_train_steps=config.save_steps,
            save_top_k=-1,
            filename="step={step}",
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    trainer = L.Trainer(
        max_steps=total_steps,
        precision=_precision_str(config.precision),
        accumulate_grad_batches=config.gradient_accumulation_steps,
        gradient_clip_val=config.max_grad_norm if config.max_grad_norm > 0 else None,
        val_check_interval=config.eval_steps if eval_path else None,
        limit_val_batches=config.limit_val_batches or 1.0,
        log_every_n_steps=config.logging_steps,
        default_root_dir=config.output_dir,
        callbacks=callbacks,
        logger=loggers,
    )

    trainer.fit(lit_module, data_module, ckpt_path=resume_from)

    import json as _json

    enc_cfg = model.encoder_config
    if hasattr(enc_cfg, "to_json"):
        enc_cfg = _json.loads(enc_cfg.to_json())
    final_path = Path(config.output_dir) / "model_final.safetensors"
    model.save_pretrained(str(final_path), encoder_config=enc_cfg)
    log.info("Training complete. Final bundle: %s", final_path)


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Train a MidiGPT model.")
    parser.add_argument("--config", required=True, help="Path to TrainConfig JSON/YAML")
    parser.add_argument("--train-data", required=True, help="Parquet shard(s): path, list, or glob")
    parser.add_argument("--eval-data", default=None, help="Optional eval parquet shard")
    parser.add_argument("--output-dir", default=None, help="Override output_dir from config")
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=None,
        help="Override gradient_accumulation_steps from config. Use with multi-GPU "
        "(DDP) so per_device_batch x grad_accum x num_gpus keeps the effective batch.",
    )
    parser.add_argument(
        "--resume-from",
        default=None,
        help="Path to a .ckpt to resume from. Use 'auto' to pick the latest "
        "checkpoint in <output-dir>/checkpoints.",
    )
    args = parser.parse_args()

    cfg = TrainConfig.from_file(args.config)
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.grad_accum is not None:
        cfg.gradient_accumulation_steps = args.grad_accum

    resume_from = args.resume_from
    if resume_from == "auto":
        ckpt_dir = Path(cfg.output_dir) / "checkpoints"
        ckpts = sorted(
            ckpt_dir.glob("*.ckpt"),
            key=lambda p: p.stat().st_mtime,
        )
        resume_from = str(ckpts[-1]) if ckpts else None
        logging.getLogger(__name__).info("Resuming from %s", resume_from or "(none found)")

    train(cfg, args.train_data, args.eval_data, resume_from=resume_from)
