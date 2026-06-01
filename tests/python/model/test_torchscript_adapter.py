"""Tests for ``midigpt.inference.model.torchscript_adapter.TorchScriptAdapter``.

Covers section 3.12 of TEST_IMPLEMENTATION_PLAN.md:

- Adapter loads a TorchScript ``.pt`` artifact from disk.
- Forward is compatible with the engine signature
  ``(input_ids, past_kv) -> (logits, present_kv)``.
- Outputs match a corresponding eager model within floating point tolerance.
- Probe-based construction (no ``ts_config``) succeeds on a GPT-2-shaped
  scripted module, and ``ModelBase`` surface methods behave sensibly.
- Probe-failure path raises ``RuntimeError`` with the expected message.
- Construction with explicit ``ts_config`` skips the probe.
- Loading a missing or corrupt artifact raises a clear error.
"""
import pathlib
from typing import List, Optional, Tuple

import pytest
import torch
from torch import nn

from midigpt.inference.model.torchscript_adapter import TorchScriptAdapter


VOCAB = 32
N_POS = 64


# --------------------------------------------------------------------------- #
#  Helper modules
# --------------------------------------------------------------------------- #
class _ProbeShapedModule(nn.Module):
    """GPT-2-shaped module whose ``forward(ids, past_kv)`` accepts the probe.

    Required attributes for ``TorchScriptAdapter._probe``:
      - ``transformer.wte`` (Embedding) — yields ``n_embd``
      - ``transformer.h`` (ModuleList of layers) — yields ``n_layer``
      - ``transformer.wpe`` (Embedding) — yields ``n_positions``

    The forward is deterministic so we can compare eager vs scripted outputs.
    """

    def __init__(self, vocab: int = VOCAB, n_embd: int = 16, n_layer: int = 2,
                 n_pos: int = N_POS):
        super().__init__()
        self.transformer = nn.Module()
        self.transformer.wte = nn.Embedding(vocab, n_embd)
        self.transformer.wpe = nn.Embedding(n_pos, n_embd)
        self.transformer.h = nn.ModuleList([nn.Identity() for _ in range(n_layer)])
        self.head = nn.Linear(n_embd, vocab, bias=False)
        # Pin weights so scripted and eager outputs are equal.
        torch.manual_seed(0)
        nn.init.normal_(self.transformer.wte.weight, std=0.02)
        nn.init.normal_(self.transformer.wpe.weight, std=0.02)
        nn.init.normal_(self.head.weight, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        past_kv: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        h = self.transformer.wte(input_ids)
        logits = self.head(h)
        out: List[Tuple[torch.Tensor, torch.Tensor]] = past_kv if past_kv is not None else []
        return logits, out


class _BadNEmbdModule(nn.Module):
    """Probe must fail: ``n_embd=7`` is divisible by none of (8, 16, 12, 4)."""

    vocab: int

    def __init__(self):
        super().__init__()
        self.vocab = VOCAB
        self.transformer = nn.Module()
        self.transformer.wte = nn.Embedding(VOCAB, 7)
        self.transformer.wpe = nn.Embedding(N_POS, 7)
        self.transformer.h = nn.ModuleList([nn.Identity()])

    def forward(
        self,
        input_ids: torch.Tensor,
        past_kv: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        empty: List[Tuple[torch.Tensor, torch.Tensor]] = []
        vocab = self.vocab
        return torch.zeros(input_ids.shape[0], input_ids.shape[1], vocab), empty


def _build_scripted(tmp_path: pathlib.Path,
                    module: Optional[nn.Module] = None) -> Tuple[nn.Module, pathlib.Path]:
    """Script ``module`` (or a fresh ``_ProbeShapedModule``), save to disk,
    reload, and return ``(loaded_scripted_module, artifact_path)``."""
    if module is None:
        module = _ProbeShapedModule()
    module.eval()
    scripted = torch.jit.script(module)
    path = tmp_path / "ts_model.pt"
    torch.jit.save(scripted, str(path))
    loaded = torch.jit.load(str(path))
    return loaded, path


# --------------------------------------------------------------------------- #
#  Adapter construction + probe
# --------------------------------------------------------------------------- #
def test_adapter_probe_infers_layout_from_scripted_module(tmp_path):
    loaded, _ = _build_scripted(tmp_path)
    adapter = TorchScriptAdapter(loaded)

    # Probe is meant to succeed on the FIRST candidate that both divides
    # n_embd and survives the forward call. For n_embd=16: candidates are
    # (8, 16, 12, 4); 16%8==0 and the forward succeeds, so n_head=8.
    assert adapter._cfg["n_embd"] == 16
    assert adapter._cfg["n_layer"] == 2
    assert adapter._cfg["n_positions"] == N_POS
    assert adapter._cfg["n_head"] == 8
    assert adapter._cfg["head_dim"] == 16 // 8
    assert adapter.arch == "torchscript"
    assert adapter.encoder_config is None


@pytest.fixture(scope="module")
def bad_scripted_path(tmp_path_factory):
    """Script the BadNEmbd module once — TorchScript registers classes
    in a process-wide compilation unit, so re-scripting the same class
    in the same process triggers 'Can't redefine method' errors."""
    bad = _BadNEmbdModule()
    bad.eval()
    scripted = torch.jit.script(bad)
    p = tmp_path_factory.mktemp("ts") / "bad.pt"
    torch.jit.save(scripted, str(p))
    return p


def test_adapter_skips_probe_when_ts_config_given(bad_scripted_path):
    loaded = torch.jit.load(str(bad_scripted_path))
    cfg = {"n_head": 1, "n_layer": 1, "n_embd": 7,
           "head_dim": 7, "n_positions": 123}
    adapter = TorchScriptAdapter(loaded, ts_config=cfg)
    assert adapter._cfg is cfg
    assert adapter.max_context() == 123


def test_adapter_probe_failure_raises_runtimeerror(bad_scripted_path):
    loaded = torch.jit.load(str(bad_scripted_path))
    with pytest.raises(RuntimeError) as exc:
        TorchScriptAdapter(loaded)
    msg = str(exc.value)
    assert "TorchScriptAdapter" in msg
    assert "could not infer model layout" in msg
    assert "ts_config" in msg


# --------------------------------------------------------------------------- #
#  ModelBase surface
# --------------------------------------------------------------------------- #
def test_adapter_make_empty_kv_shapes(tmp_path):
    loaded, _ = _build_scripted(tmp_path)
    adapter = TorchScriptAdapter(loaded)
    kv = adapter.make_empty_kv()

    assert isinstance(kv, tuple)
    assert len(kv) == adapter._cfg["n_layer"]
    for layer_kv in kv:
        assert isinstance(layer_kv, tuple) and len(layer_kv) == 2
        k, v = layer_kv
        assert k.shape == (1, adapter._cfg["n_head"], 0, adapter._cfg["head_dim"])
        assert v.shape == (1, adapter._cfg["n_head"], 0, adapter._cfg["head_dim"])


def test_adapter_kv_length_handles_empty_and_populated(tmp_path):
    loaded, _ = _build_scripted(tmp_path)
    adapter = TorchScriptAdapter(loaded)

    assert adapter.kv_length(None) == 0
    assert adapter.kv_length(()) == 0
    empty = adapter.make_empty_kv()
    assert adapter.kv_length(empty) == 0

    # Fake a populated KV at length 5.
    n_head = adapter._cfg["n_head"]
    head_dim = adapter._cfg["head_dim"]
    populated = tuple(
        (torch.zeros(1, n_head, 5, head_dim), torch.zeros(1, n_head, 5, head_dim))
        for _ in range(adapter._cfg["n_layer"])
    )
    assert adapter.kv_length(populated) == 5


def test_adapter_kv_null_positions_zeroes_v_and_negs_k(tmp_path):
    loaded, _ = _build_scripted(tmp_path)
    adapter = TorchScriptAdapter(loaded)
    n_head = adapter._cfg["n_head"]
    head_dim = adapter._cfg["head_dim"]
    populated = tuple(
        (torch.ones(1, n_head, 6, head_dim), torch.ones(1, n_head, 6, head_dim))
        for _ in range(adapter._cfg["n_layer"])
    )
    adapter.kv_null_positions(populated, [(1, 3)])
    for k, v in populated:
        # Touched range
        assert torch.all(k[:, :, 1:3, :] == -1e4)
        assert torch.all(v[:, :, 1:3, :] == 0.0)
        # Untouched range
        assert torch.all(k[:, :, 0:1, :] == 1.0)
        assert torch.all(v[:, :, 3:6, :] == 1.0)

    # No-op cases
    adapter.kv_null_positions(None, [(0, 1)])  # past_kv None
    adapter.kv_null_positions(populated, [])   # spans empty
    # Untouched ranges still intact.
    for k, _ in populated:
        assert torch.all(k[:, :, 0:1, :] == 1.0)


def test_adapter_max_context_returns_int_from_cfg(tmp_path):
    loaded, _ = _build_scripted(tmp_path)
    adapter = TorchScriptAdapter(loaded)
    mc = adapter.max_context()
    assert isinstance(mc, int)
    assert mc == N_POS


def test_adapter_parameters_delegates_to_scripted_module(tmp_path):
    loaded, _ = _build_scripted(tmp_path)
    adapter = TorchScriptAdapter(loaded)
    params = list(adapter.parameters())
    # Our shaped module has wte, wpe, and head — 3 weight tensors.
    assert len(params) == 3
    # All should be tensors with > 0 elements.
    for p in params:
        assert isinstance(p, torch.Tensor)
        assert p.numel() > 0


# --------------------------------------------------------------------------- #
#  Forward compatibility with engine signature
# --------------------------------------------------------------------------- #
def test_adapter_forward_engine_signature_without_past_kv(tmp_path):
    loaded, _ = _build_scripted(tmp_path)
    adapter = TorchScriptAdapter(loaded)
    ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)

    with torch.no_grad():
        logits, present = adapter.forward(ids)

    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (1, 4, VOCAB)
    # When past_kv was not passed, our shaped module returns an empty container.
    assert len(present) == 0


def test_adapter_forward_engine_signature_with_past_kv(tmp_path):
    loaded, _ = _build_scripted(tmp_path)
    adapter = TorchScriptAdapter(loaded)
    ids = torch.tensor([[7, 8]], dtype=torch.long)
    empty_kv = adapter.make_empty_kv()

    with torch.no_grad():
        logits, present = adapter.forward(ids, empty_kv)

    assert logits.shape == (1, 2, VOCAB)
    # Shaped module returns past_kv unchanged.
    assert isinstance(present, (tuple, list))
    assert len(present) == adapter._cfg["n_layer"]


def test_adapter_call_dunder_dispatches_to_scripted(tmp_path):
    loaded, _ = _build_scripted(tmp_path)
    adapter = TorchScriptAdapter(loaded)
    ids = torch.tensor([[0, 1, 2]], dtype=torch.long)
    with torch.no_grad():
        logits_call, _ = adapter(ids)
        logits_fwd, _ = adapter.forward(ids)
    assert torch.equal(logits_call, logits_fwd)


# --------------------------------------------------------------------------- #
#  Scripted output matches eager within tolerance
# --------------------------------------------------------------------------- #
def test_adapter_output_matches_eager_within_tolerance(tmp_path):
    eager = _ProbeShapedModule()
    eager.eval()
    scripted = torch.jit.script(eager)
    path = tmp_path / "ts_model.pt"
    torch.jit.save(scripted, str(path))
    loaded = torch.jit.load(str(path))

    adapter = TorchScriptAdapter(loaded)
    ids = torch.tensor([[3, 1, 4, 1, 5, 9, 2, 6]], dtype=torch.long)

    with torch.no_grad():
        eager_logits, _ = eager(ids)
        adapter_logits, _ = adapter.forward(ids)

    assert eager_logits.shape == adapter_logits.shape == (1, 8, VOCAB)
    assert torch.allclose(eager_logits, adapter_logits, atol=1e-5, rtol=1e-5)


# --------------------------------------------------------------------------- #
#  Disk I/O error cases
# --------------------------------------------------------------------------- #
def test_loading_missing_torchscript_artifact_raises(tmp_path):
    missing = tmp_path / "does_not_exist.pt"
    with pytest.raises((RuntimeError, FileNotFoundError, ValueError)) as exc:
        torch.jit.load(str(missing))
    # The error message should at least reference the path or the failure mode.
    text = str(exc.value).lower()
    assert (
        "does_not_exist" in text
        or "no such file" in text
        or "cannot" in text
        or "fail" in text
    )


def test_loading_corrupt_torchscript_artifact_raises(tmp_path):
    corrupt = tmp_path / "corrupt.pt"
    corrupt.write_bytes(b"not a valid torchscript archive\x00\x01\x02")
    with pytest.raises((RuntimeError, ValueError)):
        torch.jit.load(str(corrupt))


def test_roundtrip_save_load_then_adapter_still_works(tmp_path):
    """End-to-end: build → script → save → load → wrap → forward, all clean."""
    loaded, path = _build_scripted(tmp_path)
    assert path.exists()
    assert path.stat().st_size > 0

    adapter = TorchScriptAdapter(loaded)
    ids = torch.tensor([[10, 11, 12]], dtype=torch.long)
    with torch.no_grad():
        logits, present = adapter.forward(ids)
    assert logits.shape == (1, 3, VOCAB)
    assert isinstance(present, (tuple, list))
