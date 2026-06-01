"""Tests for `inference/model/transformer_lm_base.py` (test plan section 3.10).

Covers:
- `resolve_device` device selection & error handling.
- Packed-bundle save / load roundtrip via the concrete `GPT2LMHeadModel`
  subclass (the only concrete `TransformerLMBase` in the repo).
- Arch-mismatch and non-bundle error paths in `from_pretrained`.
- The forward / KV-cache contract documented by ``ModelBase`` and exercised
  by `TransformerLMBase` subclasses: incremental forward extends KV,
  `position_ids` are respected, and `make_empty_kv` / `kv_length` /
  `max_context` agree.
- Required-method enforcement: a subclass missing `arch` / `Config` cannot be
  loaded via the packed-bundle path.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import pytest
import torch

from midigpt.inference.model.gpt2 import GPT2Config, GPT2LMHeadModel
from midigpt.inference.model.transformer_lm_base import (
    PACKED_FORMAT_VERSION,
    TransformerLMBase,
    resolve_device,
)


# --------------------------------------------------------------------------- #
#  resolve_device
# --------------------------------------------------------------------------- #
def test_resolve_device_cpu_returns_cpu():
    assert resolve_device("cpu") == torch.device("cpu")


def test_resolve_device_accepts_torch_device_instance():
    assert resolve_device(torch.device("cpu")) == torch.device("cpu")


def test_resolve_device_none_picks_available_accelerator():
    dev = resolve_device(None)
    if torch.cuda.is_available():
        assert dev == torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        assert dev == torch.device("mps")
    else:
        assert dev == torch.device("cpu")


def test_resolve_device_auto_string_picks_available_accelerator():
    # "auto" follows the same branch as None — assert it matches None's choice.
    assert resolve_device("auto") == resolve_device(None)


def test_resolve_device_cuda_raises_when_unavailable():
    if torch.cuda.is_available():
        pytest.skip("CUDA IS available — polarity test only meaningful without it.")
    with pytest.raises(RuntimeError, match="CUDA"):
        resolve_device("cuda")


def test_resolve_device_mps_raises_when_unavailable():
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        pytest.skip("MPS IS available — polarity test only meaningful without it.")
    with pytest.raises(RuntimeError, match="MPS"):
        resolve_device("mps")


# --------------------------------------------------------------------------- #
#  Packed-bundle save_pretrained / from_pretrained roundtrip
# --------------------------------------------------------------------------- #
def test_save_pretrained_writes_packed_bundle_with_all_fields(
    tiny_gpt2, ghost_config_json, tmp_path
):
    enc = json.loads(ghost_config_json)
    path = tmp_path / "bundle.pt"
    tiny_gpt2.save_pretrained(str(path), encoder_config=enc)

    raw = torch.load(str(path), map_location="cpu", weights_only=False)
    assert isinstance(raw, dict)
    assert raw["format_version"] == PACKED_FORMAT_VERSION
    assert raw["arch"] == GPT2LMHeadModel.arch == "gpt2"
    assert raw["config"] == asdict(tiny_gpt2.cfg)
    assert raw["encoder_config"] == enc
    assert set(raw["state_dict"].keys()) == set(tiny_gpt2.state_dict().keys())
    # State-dict tensors saved on CPU.
    for _k, v in raw["state_dict"].items():
        assert v.device.type == "cpu"


def test_from_pretrained_roundtrip_matches_state_dict_and_config(
    tiny_gpt2, ghost_config_json, tmp_path
):
    enc = json.loads(ghost_config_json)
    path = tmp_path / "bundle.pt"
    tiny_gpt2.save_pretrained(str(path), encoder_config=enc)

    loaded = GPT2LMHeadModel.from_pretrained(str(path), device="cpu")
    assert isinstance(loaded, GPT2LMHeadModel)
    assert asdict(loaded.cfg) == asdict(tiny_gpt2.cfg)
    assert loaded.encoder_config == enc

    src_sd = tiny_gpt2.state_dict()
    new_sd = loaded.state_dict()
    assert set(src_sd.keys()) == set(new_sd.keys())
    for k in src_sd:
        assert torch.allclose(src_sd[k], new_sd[k]), f"tensor {k} differs after roundtrip"


def test_from_pretrained_save_pretrained_uses_self_encoder_config_when_omitted(
    tiny_gpt2, ghost_config_json, tmp_path
):
    enc = json.loads(ghost_config_json)
    tiny_gpt2.encoder_config = enc  # set on the model
    path = tmp_path / "bundle.pt"
    tiny_gpt2.save_pretrained(str(path))  # no encoder_config kwarg
    loaded = GPT2LMHeadModel.from_pretrained(str(path), device="cpu")
    assert loaded.encoder_config == enc


def test_from_pretrained_rejects_non_bundle(tmp_path):
    path = tmp_path / "junk.pt"
    torch.save({"foo": 1}, str(path))
    with pytest.raises(ValueError, match="packed bundle"):
        GPT2LMHeadModel.from_pretrained(str(path), device="cpu")


def test_from_pretrained_rejects_wrong_arch(tiny_gpt2, tmp_path):
    path = tmp_path / "wrong_arch.pt"
    torch.save(
        {
            "format_version": PACKED_FORMAT_VERSION,
            "arch": "not-gpt2",
            "config": asdict(tiny_gpt2.cfg),
            "encoder_config": None,
            "state_dict": tiny_gpt2.state_dict(),
        },
        str(path),
    )
    with pytest.raises(ValueError, match="arch"):
        GPT2LMHeadModel.from_pretrained(str(path), device="cpu")


def test_from_pretrained_rejects_wrong_format_version(tiny_gpt2, tmp_path):
    path = tmp_path / "bad_version.pt"
    torch.save(
        {
            "format_version": PACKED_FORMAT_VERSION + 99,
            "arch": GPT2LMHeadModel.arch,
            "config": asdict(tiny_gpt2.cfg),
            "encoder_config": None,
            "state_dict": tiny_gpt2.state_dict(),
        },
        str(path),
    )
    with pytest.raises(ValueError, match="packed bundle"):
        GPT2LMHeadModel.from_pretrained(str(path), device="cpu")


# --------------------------------------------------------------------------- #
#  Required class-level attributes (arch / Config) enforcement
# --------------------------------------------------------------------------- #
def test_subclass_without_arch_cannot_load_packed_bundle(tiny_gpt2, tmp_path):
    """A subclass that forgets `arch` cannot pass the arch-equality check."""

    @dataclass
    class _Cfg:
        x: int = 0

    class Incomplete(TransformerLMBase):
        # arch intentionally NOT defined
        Config = _Cfg

    path = tmp_path / "bundle.pt"
    tiny_gpt2.save_pretrained(str(path), encoder_config=None)
    # `cls.arch` lookup raises AttributeError because the ClassVar is unset.
    with pytest.raises(AttributeError):
        Incomplete.from_pretrained(str(path), device="cpu")


def test_subclass_without_config_cannot_construct_from_bundle(tiny_gpt2, tmp_path):
    """A subclass that forgets `Config` blows up at `cls.Config(**...)`."""

    class Incomplete(TransformerLMBase):
        arch = "gpt2"  # matches the bundle so we get past arch check
        # Config intentionally NOT defined

    path = tmp_path / "bundle.pt"
    tiny_gpt2.save_pretrained(str(path), encoder_config=None)
    with pytest.raises(AttributeError):
        Incomplete.from_pretrained(str(path), device="cpu")


# --------------------------------------------------------------------------- #
#  Forward / KV-cache contract (exercised on concrete subclass)
# --------------------------------------------------------------------------- #
def test_forward_returns_logits_and_kv_with_expected_shapes(tiny_gpt2):
    torch.manual_seed(0)
    cfg = tiny_gpt2.cfg
    ids = torch.randint(0, cfg.vocab_size, (1, 5))
    with torch.no_grad():
        logits, present = tiny_gpt2(ids)
    assert logits.shape == (1, 5, cfg.vocab_size)
    assert len(present) == cfg.n_layer
    for k, v in present:
        assert k.shape == (1, cfg.n_head, 5, cfg.head_dim)
        assert v.shape == (1, cfg.n_head, 5, cfg.head_dim)


def test_make_empty_kv_has_zero_length_for_every_layer(tiny_gpt2):
    cfg = tiny_gpt2.cfg
    empty = tiny_gpt2.make_empty_kv()
    assert len(empty) == cfg.n_layer
    for k, v in empty:
        assert k.shape == (1, cfg.n_head, 0, cfg.head_dim)
        assert v.shape == (1, cfg.n_head, 0, cfg.head_dim)
    assert tiny_gpt2.kv_length(empty) == 0
    assert tiny_gpt2.kv_length(None) == 0


def test_incremental_forward_extends_kv_length(tiny_gpt2):
    torch.manual_seed(0)
    cfg = tiny_gpt2.cfg
    ids1 = torch.randint(0, cfg.vocab_size, (1, 4))
    ids2 = torch.randint(0, cfg.vocab_size, (1, 3))
    with torch.no_grad():
        _, kv1 = tiny_gpt2(ids1)
        assert tiny_gpt2.kv_length(kv1) == 4
        logits2, kv2 = tiny_gpt2(ids2, past_kv=kv1)
    assert tiny_gpt2.kv_length(kv2) == 4 + 3
    assert logits2.shape == (1, 3, cfg.vocab_size)
    # Per-layer KV grew by exactly ids2.shape[1].
    for (k_old, _), (k_new, _) in zip(kv1, kv2, strict=False):
        assert k_new.shape[2] == k_old.shape[2] + ids2.shape[1]


def test_forward_respects_explicit_position_ids(tiny_gpt2):
    """Passing explicit `position_ids` overrides the default arange-from-past_len.

    Concretely: two calls with the SAME input_ids but different position_ids
    must produce different logits (positional embedding is consulted).
    """
    torch.manual_seed(0)
    cfg = tiny_gpt2.cfg
    ids = torch.randint(0, cfg.vocab_size, (1, 3))
    pos_a = torch.tensor([[0, 1, 2]])
    pos_b = torch.tensor([[10, 11, 12]])
    with torch.no_grad():
        logits_a, _ = tiny_gpt2(ids, position_ids=pos_a)
        logits_b, _ = tiny_gpt2(ids, position_ids=pos_b)
        # Sanity: default (no position_ids, no past_kv) matches pos_a == [0,1,2].
        logits_default, _ = tiny_gpt2(ids)
    assert not torch.allclose(logits_a, logits_b)
    assert torch.allclose(logits_a, logits_default)


def test_max_context_matches_config(tiny_gpt2):
    assert tiny_gpt2.max_context() == tiny_gpt2.cfg.n_positions


def test_roundtrip_preserves_forward_logits_bitwise(tiny_gpt2, ghost_config_json, tmp_path):
    """After save/load, the same input produces the same logits."""
    torch.manual_seed(0)
    cfg = tiny_gpt2.cfg
    ids = torch.randint(0, cfg.vocab_size, (1, 6))
    with torch.no_grad():
        ref_logits, _ = tiny_gpt2(ids)

    path = tmp_path / "bundle.pt"
    tiny_gpt2.save_pretrained(str(path), encoder_config=json.loads(ghost_config_json))
    loaded = GPT2LMHeadModel.from_pretrained(str(path), device="cpu")
    loaded.eval()
    with torch.no_grad():
        new_logits, _ = loaded(ids)
    assert torch.allclose(ref_logits, new_logits)


def test_real_score_token_forward_runs(tiny_gpt2, ghost_tokenizer, real_score):
    """End-to-end: encode a real MIDI, feed a window through the model,
    confirm logits shape and KV growth match the contract."""
    cfg = tiny_gpt2.cfg
    tokens = ghost_tokenizer.encode(real_score)
    assert isinstance(tokens, list) and len(tokens) > 0
    window = tokens[: min(len(tokens), cfg.n_positions, 64)]
    ids = torch.tensor([window], dtype=torch.long)
    with torch.no_grad():
        logits, kv = tiny_gpt2(ids)
    assert logits.shape == (1, ids.shape[1], cfg.vocab_size)
    assert tiny_gpt2.kv_length(kv) == ids.shape[1]
    # Token IDs must lie in the model's vocab range — they came from the same
    # tokenizer the model was sized against.
    assert int(ids.max()) < cfg.vocab_size and int(ids.min()) >= 0
