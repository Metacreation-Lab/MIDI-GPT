"""Protocol that every model architecture must satisfy."""
from __future__ import annotations
from typing import TYPE_CHECKING, Callable, ClassVar, Protocol, runtime_checkable

if TYPE_CHECKING:
    import torch


@runtime_checkable
class ModelBase(Protocol):
    arch: ClassVar[str]
    encoder_config: "dict | None"

    def forward(
        self,
        input_ids: "torch.Tensor",
        past_kv: "tuple | None" = None,
    ) -> "tuple[torch.Tensor, tuple]":
        """Return (logits, present_kv).

        logits shape: (batch, seq_len, vocab_size)
        present_kv:   architecture-defined; must be accepted back as past_kv.
        """
        ...

    def make_empty_kv(self) -> tuple:
        """Return a zero-length past_kv suitable as the first call's past_kv."""
        ...

    def max_context(self) -> int:
        """Maximum number of tokens the model can attend to (positional budget)."""
        ...

    def forward_with_hooks(
        self,
        input_ids: "torch.Tensor",
        past_kv: "tuple | None",
        hooks: "dict[str, Callable]",
    ) -> "tuple[torch.Tensor, tuple, dict]":
        """Like forward(), but also fires per-layer callbacks.

        hooks keys (all optional):
          "attn"   — called with (layer_idx, attn_weights: Tensor) after each attn block
          "hidden" — called with (layer_idx, hidden: Tensor) after each block output
          "logits" — called with (logits: Tensor) before returning

        Returns (logits, present_kv, hook_outputs) where hook_outputs maps
        each key to a list of values collected across layers.
        """
        ...

    @classmethod
    def from_pretrained(
        cls,
        path: str,
        device: "str | torch.device | None" = "cpu",
        dtype: "torch.dtype | None" = None,
    ) -> "ModelBase":
        ...

    def save_pretrained(
        self,
        path: str,
        encoder_config: "dict | None" = None,
    ) -> None:
        ...
