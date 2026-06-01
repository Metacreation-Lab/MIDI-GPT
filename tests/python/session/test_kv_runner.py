"""Tests for midigpt.inference.session._KVRunner (section 3.14)."""

from __future__ import annotations

import pytest
import torch

from midigpt.inference.session import _KVRunner


def test_is_prefill_true_before_first_forward(fake_model_factory):
    fake = fake_model_factory()
    runner = _KVRunner(fake, fake.make_empty_kv())
    assert runner.is_prefill is True


def test_forward_returns_logits_tensor(fake_model_factory):
    fake = fake_model_factory()
    runner = _KVRunner(fake, fake.make_empty_kv())
    ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    logits = runner.forward(ids)
    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (1, 3, fake.vocab_size)


def test_first_forward_consumes_initial_kv(fake_model_factory):
    fake = fake_model_factory()
    initial = fake.make_empty_kv()
    runner = _KVRunner(fake, initial)
    runner.forward(torch.tensor([[1, 2]], dtype=torch.long))
    # The first call's past_len must be 0 (empty kv was passed in).
    assert fake.calls[0]["past_len"] == 0


def test_is_prefill_false_after_first_forward(fake_model_factory):
    fake = fake_model_factory()
    runner = _KVRunner(fake, fake.make_empty_kv())
    runner.forward(torch.tensor([[1, 2]], dtype=torch.long))
    assert runner.is_prefill is False


def test_subsequent_forward_uses_growing_kv(fake_model_factory):
    fake = fake_model_factory()
    runner = _KVRunner(fake, fake.make_empty_kv())
    runner.forward(torch.tensor([[1, 2, 3]], dtype=torch.long))
    runner.forward(torch.tensor([[4]], dtype=torch.long))
    assert fake.calls[1]["past_len"] == 3


def test_key_mask_forwarded_to_model(fake_model_factory):
    fake = fake_model_factory()
    runner = _KVRunner(fake, fake.make_empty_kv())
    km = torch.ones(3, dtype=torch.bool)
    runner.forward(torch.tensor([[1, 2, 3]], dtype=torch.long), key_mask=km)
    assert "key_mask" in fake.calls[0]["kwargs"]
    assert torch.equal(fake.calls[0]["kwargs"]["key_mask"], km)


def test_position_ids_forwarded_to_model(fake_model_factory):
    fake = fake_model_factory()
    runner = _KVRunner(fake, fake.make_empty_kv())
    pos = torch.tensor([[5, 6, 7]], dtype=torch.long)
    runner.forward(torch.tensor([[1, 2, 3]], dtype=torch.long), position_ids=pos)
    assert "position_ids" in fake.calls[0]["kwargs"]
    assert torch.equal(fake.calls[0]["kwargs"]["position_ids"], pos)


def test_no_kwargs_when_none(fake_model_factory):
    fake = fake_model_factory()
    runner = _KVRunner(fake, fake.make_empty_kv())
    runner.forward(torch.tensor([[1]], dtype=torch.long))
    assert fake.calls[0]["kwargs"] == {}


def test_null_positions_noop_when_no_kv(fake_model_factory):
    fake = fake_model_factory()
    runner = _KVRunner(fake, fake.make_empty_kv())
    # Before any forward, _past_kv is None — null_positions must be a no-op.
    runner.null_positions([(0, 2)])  # must not raise


def test_null_positions_noop_when_empty_spans(fake_model_factory):
    fake = fake_model_factory()
    runner = _KVRunner(fake, fake.make_empty_kv())
    runner.forward(torch.tensor([[1, 2, 3]], dtype=torch.long))
    runner.null_positions([])  # must not raise


def test_null_positions_zeros_v_and_neg_k(fake_model_factory):
    fake = fake_model_factory()
    runner = _KVRunner(fake, fake.make_empty_kv())
    runner.forward(torch.tensor([[1, 2, 3, 4]], dtype=torch.long))
    # Fill kv with sentinels to confirm only the requested span is mutated.
    for k, v in runner._past_kv:
        k.fill_(1.0)
        v.fill_(2.0)
    runner.null_positions([(1, 3)])
    for k, v in runner._past_kv:
        assert torch.all(k[:, :, 1:3, :] == -1e4)
        assert torch.all(v[:, :, 1:3, :] == 0.0)
        assert torch.all(k[:, :, 0:1, :] == 1.0)  # untouched
        assert torch.all(v[:, :, 3:4, :] == 2.0)  # untouched


def test_forward_falls_back_to_positional_only_on_kwarg_error(fake_model_factory):
    """If the model rejects kwargs (e.g. TorchScript signature), _KVRunner
    retries with (ctx,) only and still returns logits."""

    class _StrictModel:
        arch = "strict"

        def __init__(self, vocab):
            self.vocab_size = vocab
            self.calls = []

        def __call__(self, ids, *args, **kwargs):
            if args or kwargs:
                raise TypeError("strict signature: positional ids only")
            self.calls.append(ids.shape)
            return torch.zeros(ids.shape[0], ids.shape[1], self.vocab_size)

        def make_empty_kv(self):
            return ()

        def kv_length(self, kv):
            return 0

        def kv_null_positions(self, kv, spans):
            pass

        def max_context(self):
            return 64

    m = _StrictModel(32)
    runner = _KVRunner(m, ())
    out = runner.forward(torch.tensor([[1, 2]], dtype=torch.long))
    assert out.shape == (1, 2, 32)
    assert m.calls == [(1, 2)]
