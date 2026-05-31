"""Tests for `midigpt.inference.model.gpt2` — section 3.11 of TEST_IMPLEMENTATION_PLAN."""
from __future__ import annotations

import copy
import json

import pytest
import torch

from midigpt.inference.model.gpt2 import GPT2Config, GPT2LMHeadModel


# --------------------------------------------------------------------------- #
#  GPT2Config
# --------------------------------------------------------------------------- #
def test_gpt2config_defaults_are_documented_values():
    cfg = GPT2Config()
    assert cfg.vocab_size == 647
    assert cfg.n_positions == 2048
    assert cfg.n_embd == 512
    assert cfg.n_layer == 6
    assert cfg.n_head == 8


def test_gpt2config_head_dim_is_n_embd_over_n_head():
    cfg = GPT2Config(n_embd=64, n_head=8)
    assert cfg.head_dim == 8
    cfg2 = GPT2Config(n_embd=16, n_head=2)
    assert cfg2.head_dim == 8


# --------------------------------------------------------------------------- #
#  Forward shape
# --------------------------------------------------------------------------- #
def test_forward_returns_logits_with_shape_B_T_vocab(tiny_gpt2, tiny_gpt2_config):
    torch.manual_seed(0)
    B, T = 1, 5
    ids = torch.randint(0, tiny_gpt2_config.vocab_size, (B, T))
    with torch.no_grad():
        logits, present_kv = tiny_gpt2(ids)
    assert logits.shape == (B, T, tiny_gpt2_config.vocab_size)


def test_forward_returns_present_kv_with_per_layer_shape(tiny_gpt2, tiny_gpt2_config):
    torch.manual_seed(0)
    B, T = 1, 7
    ids = torch.randint(0, tiny_gpt2_config.vocab_size, (B, T))
    with torch.no_grad():
        _, present_kv = tiny_gpt2(ids)
    assert len(present_kv) == tiny_gpt2_config.n_layer
    for k, v in present_kv:
        assert k.shape == (B, tiny_gpt2_config.n_head, T,
                           tiny_gpt2_config.head_dim)
        assert v.shape == (B, tiny_gpt2_config.n_head, T,
                           tiny_gpt2_config.head_dim)


# --------------------------------------------------------------------------- #
#  KV cache equivalence: token-by-token == one-shot
# --------------------------------------------------------------------------- #
def test_kv_cache_token_by_token_matches_one_shot(tiny_gpt2, tiny_gpt2_config):
    torch.manual_seed(0)
    T = 6
    ids = torch.randint(0, tiny_gpt2_config.vocab_size, (1, T))

    with torch.no_grad():
        full_logits, full_kv = tiny_gpt2(ids)

        kv = None
        step_logits = []
        for t in range(T):
            tok = ids[:, t : t + 1]
            lg, kv = tiny_gpt2(tok, past_kv=kv)
            step_logits.append(lg)
        stepwise_logits = torch.cat(step_logits, dim=1)

    assert stepwise_logits.shape == full_logits.shape
    assert torch.allclose(stepwise_logits, full_logits, atol=1e-5, rtol=1e-5)

    # KV grows by T after token-by-token loop and matches full pass
    assert tiny_gpt2.kv_length(kv) == T
    assert tiny_gpt2.kv_length(full_kv) == T
    for (k_a, v_a), (k_b, v_b) in zip(kv, full_kv):
        assert torch.allclose(k_a, k_b, atol=1e-5, rtol=1e-5)
        assert torch.allclose(v_a, v_b, atol=1e-5, rtol=1e-5)


def test_kv_cache_chained_forward_grows_kv_length(tiny_gpt2, tiny_gpt2_config):
    torch.manual_seed(0)
    ids1 = torch.randint(0, tiny_gpt2_config.vocab_size, (1, 4))
    ids2 = torch.randint(0, tiny_gpt2_config.vocab_size, (1, 3))
    with torch.no_grad():
        _, kv1 = tiny_gpt2(ids1)
        assert tiny_gpt2.kv_length(kv1) == 4
        _, kv2 = tiny_gpt2(ids2, past_kv=kv1)
        assert tiny_gpt2.kv_length(kv2) == 7


# --------------------------------------------------------------------------- #
#  make_empty_kv / kv_length / max_context
# --------------------------------------------------------------------------- #
def test_make_empty_kv_returns_zero_length_per_layer(tiny_gpt2, tiny_gpt2_config):
    kv = tiny_gpt2.make_empty_kv()
    assert len(kv) == tiny_gpt2_config.n_layer
    for k, v in kv:
        assert k.shape == (1, tiny_gpt2_config.n_head, 0, tiny_gpt2_config.head_dim)
        assert v.shape == (1, tiny_gpt2_config.n_head, 0, tiny_gpt2_config.head_dim)
    assert tiny_gpt2.kv_length(kv) == 0


def test_kv_length_none_and_empty(tiny_gpt2):
    assert tiny_gpt2.kv_length(None) == 0
    assert tiny_gpt2.kv_length(()) == 0


def test_max_context_equals_n_positions(tiny_gpt2, tiny_gpt2_config):
    assert tiny_gpt2.max_context() == tiny_gpt2_config.n_positions


# --------------------------------------------------------------------------- #
#  kv_null_positions
# --------------------------------------------------------------------------- #
def test_kv_null_positions_writes_negative_inf_to_K_and_zero_to_V(
    tiny_gpt2, tiny_gpt2_config
):
    torch.manual_seed(0)
    ids = torch.randint(0, tiny_gpt2_config.vocab_size, (1, 8))
    with torch.no_grad():
        _, kv = tiny_gpt2(ids)

    span = (0, 3)
    tiny_gpt2.kv_null_positions(kv, [span])
    for k, v in kv:
        s, e = span
        assert torch.all(k[:, :, s:e, :] == -1e4)
        assert torch.all(v[:, :, s:e, :] == 0.0)
        # outside span untouched (not the -1e4 sentinel)
        assert not torch.all(k[:, :, e:, :] == -1e4)


def test_kv_null_positions_noop_on_none_or_empty_spans(tiny_gpt2, tiny_gpt2_config):
    # Should not raise
    tiny_gpt2.kv_null_positions(None, [(0, 1)])
    torch.manual_seed(0)
    ids = torch.randint(0, tiny_gpt2_config.vocab_size, (1, 4))
    with torch.no_grad():
        _, kv = tiny_gpt2(ids)
    snapshot = [(k.clone(), v.clone()) for k, v in kv]
    tiny_gpt2.kv_null_positions(kv, [])
    for (k, v), (k0, v0) in zip(kv, snapshot):
        assert torch.equal(k, k0)
        assert torch.equal(v, v0)


# --------------------------------------------------------------------------- #
#  attention key_mask and position_ids
# --------------------------------------------------------------------------- #
def test_position_ids_override_changes_outputs(tiny_gpt2, tiny_gpt2_config):
    """Custom position_ids must actually feed wpe — different positions =>
    different logits."""
    torch.manual_seed(0)
    ids = torch.randint(0, tiny_gpt2_config.vocab_size, (1, 4))
    with torch.no_grad():
        logits_default, _ = tiny_gpt2(ids)
        custom_pos = torch.tensor([[10, 11, 12, 13]])
        logits_shifted, _ = tiny_gpt2(ids, position_ids=custom_pos)
        # Same as default when explicitly passing arange(0,T)
        logits_same, _ = tiny_gpt2(
            ids, position_ids=torch.arange(0, 4).unsqueeze(0)
        )

    assert not torch.allclose(logits_default, logits_shifted, atol=1e-5)
    assert torch.allclose(logits_default, logits_same, atol=1e-5, rtol=1e-5)


def test_key_mask_changes_attention_outputs(tiny_gpt2, tiny_gpt2_config):
    """A key_mask that hides some context positions must change the
    resulting logits versus an all-visible mask."""
    torch.manual_seed(0)
    T = 6
    ids = torch.randint(0, tiny_gpt2_config.vocab_size, (1, T))
    all_visible = torch.ones(T, dtype=torch.bool)
    partial = torch.ones(T, dtype=torch.bool)
    partial[1] = False
    partial[2] = False

    with torch.no_grad():
        logits_full, _ = tiny_gpt2(ids, key_mask=all_visible)
        logits_baseline, _ = tiny_gpt2(ids)
        logits_masked, _ = tiny_gpt2(ids, key_mask=partial)

    # The all-visible key_mask must match the no-mask fast path.
    assert torch.allclose(logits_full, logits_baseline, atol=1e-5, rtol=1e-5)
    # Hiding real context positions must perturb the output.
    assert not torch.allclose(logits_full, logits_masked, atol=1e-5)


# --------------------------------------------------------------------------- #
#  forward_with_hooks
# --------------------------------------------------------------------------- #
def test_forward_with_hooks_collects_attn_hidden_and_logits(
    tiny_gpt2, tiny_gpt2_config
):
    torch.manual_seed(0)
    B, T = 1, 5
    ids = torch.randint(0, tiny_gpt2_config.vocab_size, (B, T))
    seen = {"attn": [], "hidden": [], "logits": []}

    hooks = {
        "attn": lambda i, w: seen["attn"].append((i, w)),
        "hidden": lambda i, h: seen["hidden"].append((i, h)),
        "logits": lambda lg: seen["logits"].append(lg),
    }
    with torch.no_grad():
        logits, present, outs = tiny_gpt2.forward_with_hooks(ids, None, hooks)

    assert logits.shape == (B, T, tiny_gpt2_config.vocab_size)
    assert len(present) == tiny_gpt2_config.n_layer

    assert len(seen["attn"]) == tiny_gpt2_config.n_layer
    for i, (idx, w) in enumerate(seen["attn"]):
        assert idx == i
        assert w.shape == (B, tiny_gpt2_config.n_head, T, T)

    assert len(seen["hidden"]) == tiny_gpt2_config.n_layer
    for i, (idx, h) in enumerate(seen["hidden"]):
        assert idx == i
        assert h.shape == (B, T, tiny_gpt2_config.n_embd)

    assert len(seen["logits"]) == 1
    assert seen["logits"][0].shape == (B, T, tiny_gpt2_config.vocab_size)

    # hook_outputs mirror the recorded callbacks
    assert len(outs["attn"]) == tiny_gpt2_config.n_layer
    assert len(outs["hidden"]) == tiny_gpt2_config.n_layer
    assert len(outs["logits"]) == 1


# --------------------------------------------------------------------------- #
#  Packed bundle roundtrip preserves outputs bit-equal
# --------------------------------------------------------------------------- #
def test_save_then_from_pretrained_preserves_logits_bit_equal(
    packed_bundle_path, tiny_gpt2, tiny_gpt2_config
):
    loaded = GPT2LMHeadModel.from_pretrained(str(packed_bundle_path))
    loaded.eval()

    # Same config
    assert loaded.cfg.vocab_size == tiny_gpt2_config.vocab_size
    assert loaded.cfg.n_positions == tiny_gpt2_config.n_positions
    assert loaded.cfg.n_embd == tiny_gpt2_config.n_embd
    assert loaded.cfg.n_layer == tiny_gpt2_config.n_layer
    assert loaded.cfg.n_head == tiny_gpt2_config.n_head

    # Bit-equal state dicts
    sd_orig = tiny_gpt2.state_dict()
    sd_loaded = loaded.state_dict()
    assert set(sd_orig.keys()) == set(sd_loaded.keys())
    for k in sd_orig:
        assert torch.equal(sd_orig[k], sd_loaded[k]), f"tensor mismatch at {k}"

    # Bit-equal logits on same input
    torch.manual_seed(0)
    ids = torch.randint(0, tiny_gpt2_config.vocab_size, (1, 9))
    with torch.no_grad():
        lg1, _ = tiny_gpt2(ids)
        lg2, _ = loaded(ids)
    assert torch.equal(lg1, lg2)


def test_save_then_load_preserves_encoder_config(
    packed_bundle_path, ghost_config_json
):
    loaded = GPT2LMHeadModel.from_pretrained(str(packed_bundle_path))
    assert loaded.encoder_config is not None
    assert loaded.encoder_config == json.loads(ghost_config_json)


# --------------------------------------------------------------------------- #
#  Realistic end-to-end: real MIDI -> tokens -> tiny_gpt2 forward
# --------------------------------------------------------------------------- #
def test_tiny_gpt2_handles_real_midi_token_sequence(
    tiny_gpt2, tiny_gpt2_config, ghost_tokenizer, real_score
):
    """Smoke test: real tokens from a real MIDI file flow through the
    model without shape mismatches, and KV equivalence holds on real data."""
    tokens = ghost_tokenizer.encode(real_score)
    assert isinstance(tokens, list)
    assert len(tokens) > 0
    vocab = tiny_gpt2_config.vocab_size
    assert all(0 <= t < vocab for t in tokens)

    # Truncate to fit positional window with margin
    T = min(len(tokens), tiny_gpt2_config.n_positions - 1, 64)
    ids = torch.tensor(tokens[:T], dtype=torch.long).unsqueeze(0)

    torch.manual_seed(0)
    with torch.no_grad():
        full_logits, full_kv = tiny_gpt2(ids)
    assert full_logits.shape == (1, T, vocab)
    assert tiny_gpt2.kv_length(full_kv) == T

    # KV equivalence on the realistic sequence
    with torch.no_grad():
        kv = None
        step_logits = []
        for t in range(T):
            lg, kv = tiny_gpt2(ids[:, t : t + 1], past_kv=kv)
            step_logits.append(lg)
        stepwise = torch.cat(step_logits, dim=1)
    assert torch.allclose(stepwise, full_logits, atol=1e-5, rtol=1e-5)
