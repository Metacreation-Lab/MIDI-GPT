"""Model architecture registry and built-in architectures."""

from midigpt.inference.model.gpt2 import GPT2Config, GPT2LMHeadModel
from midigpt.inference.model.registry import REGISTRY, get_model_class, register
from midigpt.inference.model.torchscript_adapter import TorchScriptAdapter
from midigpt.inference.model.transformer_lm_base import (
    TransformerLMBase,
    resolve_device,
)

__all__ = [
    "REGISTRY",
    "GPT2Config",
    "GPT2LMHeadModel",
    "TorchScriptAdapter",
    "TransformerLMBase",
    "get_model_class",
    "register",
    "resolve_device",
]
