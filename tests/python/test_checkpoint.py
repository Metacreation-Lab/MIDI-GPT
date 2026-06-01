"""Tests for ``midigpt.tokenizer.checkpoint`` (plan section 3.5).

Covers ``load_checkpoint`` for the two supported on-disk layouts:

  * Packed ``.pt`` bundle (new): a single file containing
    ``format_version``, ``arch``, ``config``, ``encoder_config`` and
    ``state_dict``.  Loaded into a ready-to-use ``nn.Module``.
  * Legacy directory layout: ``config.json`` (encoder spec) + ``model.pt``
    (TorchScript module).  The loader does NOT instantiate the TorchScript
    model itself — it returns the ``model_path`` so a downstream engine /
    adapter can do that.

Plus error paths: missing files, malformed ``.pt`` files, missing
``encoder_config`` in a packed bundle, and unsupported path kinds.
"""
from __future__ import annotations

import pathlib

import pytest
import torch

import midigpt._core as _core
from midigpt.attributes.base import AttributeAnalyzer
from midigpt.tokenizer.checkpoint import CheckpointBundle, load_checkpoint
from midigpt.tokenizer.tokenizer import Tokenizer


def _vocab_size(enc_cfg: _core.EncoderConfig) -> int:
    """Authoritative vocab size: built through the tokenizer (which is
    what the production code does)."""
    analyzer = AttributeAnalyzer.from_config(enc_cfg)
    return Tokenizer(enc_cfg, analyzer).vocab_size()


# --------------------------------------------------------------------------- #
#  Packed .pt bundle
# --------------------------------------------------------------------------- #
def test_load_checkpoint_packed_bundle_returns_ready_model(packed_bundle_path):
    bundle = load_checkpoint(str(packed_bundle_path))

    assert isinstance(bundle, CheckpointBundle)
    assert bundle.model is not None, "packed bundle should hydrate the model"
    assert bundle.model_path is None, (
        "packed bundle uses .model, not .model_path"
    )
    assert isinstance(bundle.encoder_config, _core.EncoderConfig)
    # The hydrated model exposes its packed encoder_config, and our loader
    # uses it to build the EncoderConfig instance — sanity-check the vocab
    # size matches what the model was built for.
    assert bundle.model.cfg.vocab_size > 0
    assert _vocab_size(bundle.encoder_config) == bundle.model.cfg.vocab_size


def test_load_checkpoint_packed_bundle_format_version_present(packed_bundle_path):
    """The on-disk bundle must carry ``format_version`` — that is the gate
    distinguishing packed bundles from random pickled objects."""
    raw = torch.load(str(packed_bundle_path), map_location="cpu", weights_only=False)
    assert isinstance(raw, dict)
    assert "format_version" in raw
    assert raw["format_version"] == 1
    assert "state_dict" in raw
    assert "encoder_config" in raw
    assert raw["arch"] == "gpt2"


# --------------------------------------------------------------------------- #
#  Legacy directory layout (config.json + model.pt)
# --------------------------------------------------------------------------- #
def _write_legacy_dir(dir_path: pathlib.Path, config_json: str) -> pathlib.Path:
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / "config.json").write_text(config_json)

    # `load_checkpoint` for the directory layout only records the
    # `model.pt` path; nothing in the loader itself reads it.  We still
    # save a *real* TorchScript file so the "downstream TorchScript path"
    # is exercised end-to-end (load_checkpoint -> torch.jit.load).
    class _Stub(torch.nn.Module):
        def forward(self, ids: torch.Tensor) -> torch.Tensor:
            return ids.float().sum(dim=-1, keepdim=True)

    scripted = torch.jit.script(_Stub())
    model_pt = dir_path / "model.pt"
    scripted.save(str(model_pt))
    return model_pt


def test_load_checkpoint_legacy_directory_returns_model_path(
    tmp_path, ghost_config_json,
):
    bundle_dir = tmp_path / "legacy_ckpt"
    expected_model_pt = _write_legacy_dir(bundle_dir, ghost_config_json)

    bundle = load_checkpoint(str(bundle_dir))

    assert isinstance(bundle, CheckpointBundle)
    assert bundle.model is None, (
        "legacy directory layout populates model_path, not model"
    )
    assert bundle.model_path == str(expected_model_pt)
    assert isinstance(bundle.encoder_config, _core.EncoderConfig)
    # The encoder_config parsed from disk must match the JSON we wrote.
    expected = _core.EncoderConfig.from_json(ghost_config_json)
    assert _vocab_size(bundle.encoder_config) == _vocab_size(expected)
    assert bundle.encoder_config.resolution == expected.resolution
    assert bundle.encoder_config.model_dim == expected.model_dim


def test_load_checkpoint_legacy_directory_model_pt_is_loadable_torchscript(
    tmp_path, ghost_config_json,
):
    """The directory loader's contract: ``model.pt`` is a TorchScript file
    that some downstream consumer will ``torch.jit.load``.  Prove the file
    we wrote actually loads back — this exercises the TorchScript fallback
    path the loader is designed to feed."""
    bundle_dir = tmp_path / "legacy_ts"
    _write_legacy_dir(bundle_dir, ghost_config_json)

    bundle = load_checkpoint(str(bundle_dir))
    loaded = torch.jit.load(bundle.model_path)
    out = loaded(torch.tensor([[1, 2, 3]], dtype=torch.long))
    assert out.shape == (1, 1)
    assert float(out.item()) == 6.0


# --------------------------------------------------------------------------- #
#  Error paths
# --------------------------------------------------------------------------- #
def test_load_checkpoint_missing_config_json_raises(tmp_path):
    d = tmp_path / "no_config"
    d.mkdir()
    # only model.pt, no config.json
    (d / "model.pt").write_bytes(b"\x00")
    with pytest.raises(FileNotFoundError, match="config.json"):
        load_checkpoint(str(d))


def test_load_checkpoint_missing_model_pt_raises(tmp_path, ghost_config_json):
    d = tmp_path / "no_model"
    d.mkdir()
    (d / "config.json").write_text(ghost_config_json)
    with pytest.raises(FileNotFoundError, match="model.pt"):
        load_checkpoint(str(d))


def test_load_checkpoint_pt_without_format_version_raises(tmp_path):
    bad = tmp_path / "not_a_bundle.pt"
    torch.save({"foo": 1, "bar": 2}, str(bad))
    with pytest.raises(ValueError, match="packed bundle"):
        load_checkpoint(str(bad))


def test_load_checkpoint_pt_bundle_with_none_encoder_config_raises(
    tmp_path, tiny_gpt2,
):
    # Save a real packed bundle, then re-save with encoder_config=None.
    raw_path = tmp_path / "no_enc.pt"
    tiny_gpt2.save_pretrained(str(raw_path), encoder_config={})
    raw = torch.load(str(raw_path), map_location="cpu", weights_only=False)
    raw["encoder_config"] = None
    torch.save(raw, str(raw_path))

    with pytest.raises(ValueError, match="encoder_config"):
        load_checkpoint(str(raw_path))


def test_load_checkpoint_random_path_raises(tmp_path):
    # Not a directory, not a .pt file — just some random non-existent path.
    bogus = tmp_path / "does_not_exist.bin"
    with pytest.raises(ValueError, match="directory or a .pt bundle"):
        load_checkpoint(str(bogus))


def test_load_checkpoint_non_pt_extension_file_raises(tmp_path):
    f = tmp_path / "weights.bin"
    f.write_bytes(b"\x00\x01\x02")
    with pytest.raises(ValueError, match="directory or a .pt bundle"):
        load_checkpoint(str(f))
