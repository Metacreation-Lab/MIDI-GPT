"""InferenceEngine.from_pretrained against a mocked Hub: the token is passed through, a cached copy is
used when the Hub can't be reached or refuses, and a gated repo with nothing cached says what to do."""

from __future__ import annotations

import httpx
import huggingface_hub
import pytest
from huggingface_hub import HfApi
from huggingface_hub.errors import GatedRepoError

from midigpt.inference.engine import InferenceEngine

REPO = "Metacreation/MIDI-GPT"


def _gated() -> GatedRepoError:
    request = httpx.Request("GET", f"https://huggingface.co/api/models/{REPO}")
    return GatedRepoError("401 gated", response=httpx.Response(401, request=request))


@pytest.fixture
def hub(monkeypatch, tmp_path):
    """Empty HF cache in tmp_path; from_checkpoint returns the path it was given; records Hub calls."""
    calls = {}
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(tmp_path))
    monkeypatch.setattr(InferenceEngine, "from_checkpoint", classmethod(lambda cls, path, **kw: path))

    def cache(*names, snapshot="abc123"):
        d = tmp_path / f"models--{REPO.replace('/', '--')}" / "snapshots" / snapshot
        d.mkdir(parents=True, exist_ok=True)
        for n in names:
            (d / n).touch()
        return d

    calls["cache"] = cache
    return calls


def _hub_raises(monkeypatch, exc):
    def fail(*a, **kw):
        raise exc
    monkeypatch.setattr(HfApi, "list_repo_files", fail)


def test_token_is_passed_to_the_hub(hub, monkeypatch):
    def list_files(self, repo_id, token=None, **kw):
        hub["list_token"] = token
        return ["yellow_medium-final.safetensors"]

    def download(repo_id, filename, token=None, **kw):
        hub["download_token"] = token
        return f"/hub/{filename}"

    monkeypatch.setattr(HfApi, "list_repo_files", list_files)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    assert InferenceEngine.from_pretrained("yellow_medium", token="hf_abc") == "/hub/yellow_medium-final.safetensors"
    assert hub["list_token"] == hub["download_token"] == "hf_abc"


def test_gated_without_cache_says_what_to_do(hub, monkeypatch):
    _hub_raises(monkeypatch, _gated())
    with pytest.raises(PermissionError, match=r"huggingface\.co/Metacreation/MIDI-GPT.*hf auth login"):
        InferenceEngine.from_pretrained("yellow_medium")


@pytest.mark.parametrize("exc", [_gated(), httpx.ConnectError("offline")])
def test_uses_the_cached_checkpoint_when_the_hub_fails(hub, monkeypatch, exc):
    d = hub["cache"]("yellow_medium-step1000.safetensors", "yellow_medium-final.safetensors",
                     "prism_medium-final.safetensors")
    _hub_raises(monkeypatch, exc)
    assert InferenceEngine.from_pretrained("yellow_medium") == str(d / "yellow_medium-final.safetensors")


def test_cached_fallback_prefers_the_main_revision(hub, monkeypatch, tmp_path):
    hub["cache"]("yellow_medium-final.safetensors", snapshot="old")
    current = hub["cache"]("yellow_medium-final.safetensors", snapshot="current")
    hub["cache"]("yellow_medium-final.safetensors", snapshot="old")   # touched last, so newest
    refs = tmp_path / f"models--{REPO.replace('/', '--')}" / "refs"
    refs.mkdir()
    (refs / "main").write_text("current")
    _hub_raises(monkeypatch, httpx.ConnectError("offline"))
    assert InferenceEngine.from_pretrained("yellow_medium") == str(current / "yellow_medium-final.safetensors")


def test_cached_fallback_picks_the_highest_step(hub, monkeypatch):
    d = hub["cache"]("prism_medium-step900.safetensors", "prism_medium-step12000.safetensors")
    _hub_raises(monkeypatch, httpx.ConnectError("offline"))
    assert InferenceEngine.from_pretrained("prism_medium") == str(d / "prism_medium-step12000.safetensors")


def test_other_errors_without_cache_are_raised_unchanged(hub, monkeypatch):
    _hub_raises(monkeypatch, httpx.ConnectError("offline"))
    with pytest.raises(httpx.ConnectError):
        InferenceEngine.from_pretrained("yellow_medium")
