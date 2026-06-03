"""Shared base class for transformer LMs with packed-checkpoint I/O.

Concrete architectures (gpt2.py, future llama.py, …) subclass
``TransformerLMBase`` and supply:

    arch:   ClassVar[str]            # registry key, e.g. "gpt2"
    Config: ClassVar[type]           # the @dataclass config used by __init__

The class then provides generic ``from_pretrained`` / ``save_pretrained`` using
the packed bundle format:

    {
      "format_version": PACKED_FORMAT_VERSION,
      "arch":           <cls.arch>,
      "config":         asdict(cfg),
      "encoder_config": {...},
      "state_dict":     {...},
    }
"""

from __future__ import annotations

import json as _json
import pathlib
from dataclasses import asdict
from typing import ClassVar

import torch
import torch.nn as nn

PACKED_FORMAT_VERSION = 1          # legacy .pt pickle format (read-only, backward compat)
SAFETENSORS_FORMAT_VERSION = 2     # current .safetensors format


def resolve_device(device: str | torch.device | None) -> torch.device:
    if device is None or device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    dev = torch.device(device) if not isinstance(device, torch.device) else device
    if dev.type == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError(
                "device='mps' requested but MPS is unavailable "
                "(requires Apple Silicon + macOS 12.3+ + PyTorch with MPS support)"
            )
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device='cuda' requested but CUDA is unavailable")
    return dev


class TransformerLMBase(nn.Module):
    """Abstract base for transformer LMs sharing the packed checkpoint format.

    Subclasses must define:
      arch:   class-level registry key
      Config: dataclass type used to construct the model (cls(cfg))
    """

    arch: ClassVar[str]
    Config: ClassVar[type]

    encoder_config: dict | None

    @classmethod
    def from_pretrained(
        cls,
        path: str,
        device: str | torch.device | None = "cpu",
        dtype: torch.dtype | None = None,
    ) -> TransformerLMBase:
        dev = resolve_device(device)
        p = pathlib.Path(path)
        if p.suffix == ".safetensors":
            return cls._from_safetensors(path, dev, dtype)
        return cls._from_pt_bundle(path, dev, dtype)

    @classmethod
    def _from_safetensors(
        cls, path: str, dev: torch.device, dtype: torch.dtype | None
    ) -> TransformerLMBase:
        try:
            from safetensors import safe_open
            from safetensors.torch import load_file
        except ImportError:
            raise ImportError("pip install midigpt[inference] to enable safetensors") from None

        with safe_open(path, framework="pt") as f:
            meta = f.metadata()

        fv = meta.get("format_version")
        if fv != str(SAFETENSORS_FORMAT_VERSION):
            raise ValueError(
                f"{path} has format_version={fv!r}; "
                f"expected {SAFETENSORS_FORMAT_VERSION!r}. "
                "Re-export with save_pretrained()."
            )
        arch = meta.get("arch", "")
        if arch != cls.arch:
            raise ValueError(f"{path} has arch={arch!r}, but {cls.__name__}.arch={cls.arch!r}")

        cfg = cls.Config(**_json.loads(meta["config"]))
        model = cls(cfg)
        model.load_state_dict(load_file(path, device="cpu"), strict=True)
        model.encoder_config = _json.loads(meta.get("encoder_config", "null"))
        model.eval()
        if dtype is not None:
            model = model.to(dtype=dtype)
        return model.to(dev)

    @classmethod
    def _from_pt_bundle(
        cls, path: str, dev: torch.device, dtype: torch.dtype | None
    ) -> TransformerLMBase:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if not (isinstance(ckpt, dict) and ckpt.get("format_version") == PACKED_FORMAT_VERSION):
            raise ValueError(
                f"{path} is not a packed bundle. "
                "Convert old checkpoints first with save_pretrained()."
            )
        arch = ckpt.get("arch") or "gpt2"
        if arch != cls.arch:
            raise ValueError(f"{path} has arch={arch!r}, but {cls.__name__}.arch={cls.arch!r}")
        enc_cfg = ckpt.get("encoder_config")
        if enc_cfg is None:
            raise ValueError(f"{path} missing 'encoder_config' — cannot tokenize without it")
        cfg = cls.Config(**ckpt["config"])
        model = cls(cfg)
        model.load_state_dict(ckpt["state_dict"], strict=True)
        model.encoder_config = enc_cfg
        model.eval()
        if dtype is not None:
            model = model.to(dtype=dtype)
        return model.to(dev)

    def save_pretrained(
        self,
        path: str,
        encoder_config: dict | None = None,
    ) -> None:
        try:
            from safetensors.torch import save_file
        except ImportError:
            raise ImportError("pip install midigpt[inference] to enable safetensors") from None

        if encoder_config is None:
            encoder_config = self.encoder_config
        metadata = {
            "format_version": str(SAFETENSORS_FORMAT_VERSION),
            "arch": self.arch,
            "config": _json.dumps(asdict(self.cfg)),
            "encoder_config": _json.dumps(encoder_config) if encoder_config is not None else "null",
        }
        state_dict = {k: v.detach().cpu().contiguous() for k, v in self.state_dict().items()}
        save_file(state_dict, path, metadata=metadata)
