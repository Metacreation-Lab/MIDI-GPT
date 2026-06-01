"""Model architecture registry and built-in architectures."""
from midigpt.inference.model.registry import register, get_model_class, REGISTRY
from midigpt.inference.model.transformer_lm_base import (
    TransformerLMBase,
    resolve_device,
)
from midigpt.inference.model.gpt2 import GPT2Config, GPT2LMHeadModel
from midigpt.inference.model.torchscript_adapter import TorchScriptAdapter

__all__ = [
    "register",
    "get_model_class",
    "REGISTRY",
    "TransformerLMBase",
    "GPT2Config",
    "GPT2LMHeadModel",
    "TorchScriptAdapter",
    "resolve_device",
]
