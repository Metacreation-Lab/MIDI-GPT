"""Model architecture registry and built-in architectures."""
from midigpt.inference.model.registry import register, get_model_class, REGISTRY
from midigpt.inference.model.gpt2 import GPT2Config, GPT2LMHeadModel, resolve_device

__all__ = [
    "register",
    "get_model_class",
    "REGISTRY",
    "GPT2Config",
    "GPT2LMHeadModel",
    "resolve_device",
]
