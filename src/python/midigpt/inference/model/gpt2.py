"""Pure-PyTorch GPT-2 — state-dict layout matches HuggingFace GPT2LMHeadModel.

State-dict layout:
    transformer.wte.weight             (V, D)
    transformer.wpe.weight             (P, D)
    transformer.h.{i}.ln_1.{weight,bias}
    transformer.h.{i}.attn.c_attn.{weight,bias}   (D, 3D), (3D,)
    transformer.h.{i}.attn.c_proj.{weight,bias}   (D, D),  (D,)
    transformer.h.{i}.ln_2.{weight,bias}
    transformer.h.{i}.mlp.c_fc.{weight,bias}      (D, 4D), (4D,)
    transformer.h.{i}.mlp.c_proj.{weight,bias}    (4D, D), (D,)
    transformer.ln_f.{weight,bias}
    lm_head.weight                                (V, D)

Attention uses F.scaled_dot_product_attention; legacy attn.bias / attn.masked_bias
buffers are stripped on load for backward compat.

Packed checkpoint format (format_version 1):
    {
      "format_version": 1,
      "arch":           "gpt2",
      "config":         {...},
      "encoder_config": {...},
      "state_dict":     {...},
    }
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from midigpt.inference.model.registry import register
from midigpt.inference.model.transformer_lm_base import TransformerLMBase


# --------------------------------------------------------------------------- #
#  Config
# --------------------------------------------------------------------------- #
@dataclass
class GPT2Config:
    vocab_size: int = 647
    n_positions: int = 2048
    n_embd: int = 512
    n_layer: int = 6
    n_head: int = 8

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head


# --------------------------------------------------------------------------- #
#  Layers
# --------------------------------------------------------------------------- #
class Conv1D(nn.Module):
    """HF GPT-2 Conv1D: weight (nx, nf), applied as x @ w + b."""

    def __init__(self, nx: int, nf: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(nx, nf))
        nn.init.normal_(self.weight, std=0.02)
        self.bias = nn.Parameter(torch.zeros(nf))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.addmm(self.bias, x.reshape(-1, x.shape[-1]), self.weight).view(
            *x.shape[:-1], self.weight.shape[1]
        )


def gelu_new(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))


class GPT2Attention(nn.Module):
    def __init__(self, cfg: GPT2Config):
        super().__init__()
        self.n_head = cfg.n_head
        self.head_dim = cfg.head_dim
        self.c_attn = Conv1D(cfg.n_embd, 3 * cfg.n_embd)
        self.c_proj = Conv1D(cfg.n_embd, cfg.n_embd)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        return x.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, _, T, _ = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, self.n_head * self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        return_attn_weights: bool = False,
        key_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor], torch.Tensor | None]:
        qkv = self.c_attn(x)
        q, k, v = qkv.split(qkv.shape[-1] // 3, dim=-1)
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)
        if past_kv is not None and past_kv[0].shape[2] > 0:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)
        present = (k, v)

        T_q, T_k = q.shape[2], k.shape[2]

        # key_mask: bool tensor (T_k,) — True = visible key position. Built from
        # encoder hidden_spans; broadcast across queries and combined with the
        # causal triangular mask. None = no span masking (cheap fast path).
        def _causal_bool_mask() -> torch.Tensor:
            return torch.ones(T_q, T_k, dtype=torch.bool, device=q.device).tril(diagonal=T_k - T_q)

        if return_attn_weights:
            # Manual attention to capture weights
            scale = self.head_dim**-0.5
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            if T_q == T_k:
                allow = torch.ones(T_q, T_k, dtype=torch.bool, device=q.device).tril()
            elif T_q > 1:
                allow = _causal_bool_mask()
            else:
                allow = torch.ones(T_q, T_k, dtype=torch.bool, device=q.device)
            if key_mask is not None:
                allow = allow & key_mask.view(1, T_k).to(allow.device)
            scores = scores.masked_fill(~allow, float("-inf"))
            attn_weights = scores.softmax(-1)
            y = torch.matmul(attn_weights, v)
        else:
            attn_weights = None
            if key_mask is None:
                if T_q == T_k:
                    y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
                elif T_q == 1:
                    y = F.scaled_dot_product_attention(q, k, v)
                else:
                    mask = _causal_bool_mask()
                    y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
            else:
                if T_q == 1:
                    allow = key_mask.view(1, T_k).to(q.device)
                else:
                    allow = _causal_bool_mask() & key_mask.view(1, T_k).to(q.device)
                y = F.scaled_dot_product_attention(q, k, v, attn_mask=allow)

        y = self._merge_heads(y)
        return self.c_proj(y), present, attn_weights


class GPT2MLP(nn.Module):
    def __init__(self, cfg: GPT2Config):
        super().__init__()
        self.c_fc = Conv1D(cfg.n_embd, 4 * cfg.n_embd)
        self.c_proj = Conv1D(4 * cfg.n_embd, cfg.n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(gelu_new(self.c_fc(x)))


class GPT2Block(nn.Module):
    def __init__(self, cfg: GPT2Config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd, eps=1e-5)
        self.attn = GPT2Attention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd, eps=1e-5)
        self.mlp = GPT2MLP(cfg)

    def forward(
        self,
        x: torch.Tensor,
        past_kv: tuple | None = None,
        return_attn_weights: bool = False,
        key_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple, torch.Tensor | None]:
        a, present, attn_w = self.attn(
            self.ln_1(x),
            past_kv=past_kv,
            return_attn_weights=return_attn_weights,
            key_mask=key_mask,
        )
        x = x + a
        x = x + self.mlp(self.ln_2(x))
        return x, present, attn_w


class GPT2Transformer(nn.Module):
    def __init__(self, cfg: GPT2Config):
        super().__init__()
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.n_positions, cfg.n_embd)
        self.drop = nn.Identity()
        self.h = nn.ModuleList([GPT2Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd, eps=1e-5)


# --------------------------------------------------------------------------- #
#  Top-level model
# --------------------------------------------------------------------------- #
@register("gpt2")
class GPT2LMHeadModel(TransformerLMBase):
    arch = "gpt2"
    Config = GPT2Config

    def __init__(self, cfg: GPT2Config):
        super().__init__()
        self.cfg = cfg
        self.encoder_config: dict | None = None
        self.transformer = GPT2Transformer(cfg)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

    # ------------------------------------------------------------------- #
    #  ModelBase interface
    # ------------------------------------------------------------------- #
    def forward(
        self,
        input_ids: torch.Tensor,
        past_kv: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None = None,
        key_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple]:
        B, T = input_ids.shape
        past_len = (
            past_kv[0][0].shape[2] if past_kv is not None and past_kv[0][0].shape[2] > 0 else 0
        )
        if position_ids is None:
            pos = torch.arange(past_len, past_len + T, device=input_ids.device).unsqueeze(0)
        else:
            pos = position_ids.to(input_ids.device)

        x = self.transformer.wte(input_ids) + self.transformer.wpe(pos)
        presents = []
        for i, block in enumerate(self.transformer.h):
            pkv = past_kv[i] if past_kv is not None else None
            x, present, _ = block(
                x,
                past_kv=pkv,
                return_attn_weights=False,
                key_mask=key_mask,
            )
            presents.append(present)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return logits, tuple(presents)

    def make_empty_kv(self) -> tuple:
        """Return zero-length past_kv for the first forward call."""
        cfg = self.cfg
        return tuple(
            (
                torch.zeros(1, cfg.n_head, 0, cfg.head_dim),
                torch.zeros(1, cfg.n_head, 0, cfg.head_dim),
            )
            for _ in range(cfg.n_layer)
        )

    def kv_length(self, past_kv) -> int:
        if past_kv is None or len(past_kv) == 0:
            return 0
        return int(past_kv[0][0].shape[2])

    def kv_null_positions(
        self,
        past_kv,
        spans: list[tuple[int, int]],
    ) -> None:
        if past_kv is None or not spans:
            return
        for k_c, v_c in past_kv:
            for s, e in spans:
                k_c[:, :, s:e, :] = -1e4
                v_c[:, :, s:e, :] = 0.0

    def max_context(self) -> int:
        return self.cfg.n_positions

    def forward_with_hooks(
        self,
        input_ids: torch.Tensor,
        past_kv: tuple | None,
        hooks: dict[str, Callable],
    ) -> tuple[torch.Tensor, tuple, dict]:
        """Forward pass that fires optional per-layer callbacks.

        hooks keys (all optional):
          "attn"   — fn(layer_idx: int, weights: Tensor)  shape (B, n_head, T_q, T_k)
          "hidden" — fn(layer_idx: int, hidden: Tensor)   shape (B, T, D)
          "logits" — fn(logits: Tensor)                   shape (B, T, V)
        """
        B, T = input_ids.shape
        past_len = (
            past_kv[0][0].shape[2] if past_kv is not None and past_kv[0][0].shape[2] > 0 else 0
        )
        pos = torch.arange(past_len, past_len + T, device=input_ids.device).unsqueeze(0)

        x = self.transformer.wte(input_ids) + self.transformer.wpe(pos)
        presents = []
        hook_outputs: dict[str, list] = {k: [] for k in hooks}
        want_attn = "attn" in hooks
        want_hidden = "hidden" in hooks

        for i, block in enumerate(self.transformer.h):
            pkv = past_kv[i] if past_kv is not None else None
            x, present, attn_w = block(x, past_kv=pkv, return_attn_weights=want_attn)
            presents.append(present)
            if want_attn and attn_w is not None:
                hooks["attn"](i, attn_w)
                hook_outputs["attn"].append(attn_w)
            if want_hidden:
                hooks["hidden"](i, x)
                hook_outputs["hidden"].append(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        if "logits" in hooks:
            hooks["logits"](logits)
            hook_outputs["logits"].append(logits)

        return logits, tuple(presents), hook_outputs

    # Checkpoint I/O (from_pretrained / save_pretrained) is inherited from
    # TransformerLMBase — the packed bundle format is architecture-agnostic.
