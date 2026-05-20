"""Architecture registry: maps arch identifier strings to model classes."""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from midigpt.inference.base import ModelBase

REGISTRY: dict[str, type] = {}


def register(arch_id: str):
    """Class decorator that adds a model class to the registry.

    Usage::

        @register("gpt2")
        class GPT2LMHeadModel(nn.Module):
            arch = "gpt2"
            ...
    """
    def decorator(cls):
        REGISTRY[arch_id] = cls
        return cls
    return decorator


def get_model_class(arch_id: str) -> "type[ModelBase]":
    if arch_id not in REGISTRY:
        raise ValueError(
            f"Unknown architecture '{arch_id}'. "
            f"Registered: {sorted(REGISTRY)}"
        )
    return REGISTRY[arch_id]
