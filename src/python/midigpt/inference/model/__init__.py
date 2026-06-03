"""Model architecture registry and built-in architectures."""

from midigpt.inference.model.gpt2 import GPT2Config, GPT2LMHeadModel
from midigpt.inference.model.registry import REGISTRY, get_model_class, register
from midigpt.inference.model.torchscript_adapter import TorchScriptAdapter
from midigpt.inference.model.transformer_lm_base import (
    PACKED_FORMAT_VERSION,
    SAFETENSORS_FORMAT_VERSION,
    TransformerLMBase,
    resolve_device,
)

__all__ = [
    "PACKED_FORMAT_VERSION",
    "REGISTRY",
    "SAFETENSORS_FORMAT_VERSION",
    "GPT2Config",
    "GPT2LMHeadModel",
    "TorchScriptAdapter",
    "TransformerLMBase",
    "get_model_class",
    "register",
    "resolve_device",
]
