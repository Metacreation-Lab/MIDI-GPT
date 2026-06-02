import json
import pathlib
from dataclasses import dataclass
from typing import Any

import midigpt._core as _core


@dataclass
class CheckpointBundle:
    encoder_config: _core.EncoderConfig
    model_path: str | None = None  # legacy: TorchScript model.pt path
    model: Any | None = None  # new: ready-to-use nn.Module


def load_checkpoint(path: str, device: str | None = None) -> CheckpointBundle:
    p = pathlib.Path(path)

    if p.is_dir():
        config_path = p / "config.json"
        model_path = p / "model.pt"
        if not config_path.exists():
            raise FileNotFoundError(f"config.json missing in: {path}")
        if not model_path.exists():
            raise FileNotFoundError(f"model.pt missing in: {path}")
        return CheckpointBundle(
            encoder_config=_core.EncoderConfig.from_json(config_path.read_text()),
            model_path=str(model_path),
        )

    if p.is_file() and p.suffix == ".pt":
        return _load_bundle_file(p, device=device)

    raise ValueError(f"Checkpoint must be a directory or a .pt bundle file: {path}")


def _load_bundle_file(p: pathlib.Path, device: str | None = None) -> CheckpointBundle:
    try:
        import torch
    except ImportError:
        raise ImportError("pip install midigpt[inference]") from None

    from midigpt.inference.model import get_model_class

    ckpt = torch.load(str(p), map_location="cpu", weights_only=False)
    if not (isinstance(ckpt, dict) and "format_version" in ckpt and "state_dict" in ckpt):
        raise ValueError(
            f"{p} is not a packed bundle (format_version + state_dict missing). "
            "Convert it first with GPT2LMHeadModel.from_torchscript(...).save_pretrained(...)."
        )
    arch = ckpt.get("arch") or "gpt2"
    model_cls = get_model_class(arch)
    model = model_cls.from_pretrained(str(p), device=device or "cpu")

    enc_cfg = model.encoder_config
    if enc_cfg is None:
        raise ValueError(f"Bundle {p} missing 'encoder_config' — cannot tokenize without it")
    if isinstance(enc_cfg, dict):
        enc_cfg_json = json.dumps(enc_cfg)
    elif isinstance(enc_cfg, str):
        enc_cfg_json = enc_cfg
    else:
        raise ValueError(f"encoder_config must be a dict or JSON string, got {type(enc_cfg)}")

    return CheckpointBundle(
        encoder_config=_core.EncoderConfig.from_json(enc_cfg_json),
        model=model,
    )
