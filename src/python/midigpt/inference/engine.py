from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

from midigpt.attributes.base import AttributeAnalyzer
from midigpt.tokenizer.tokenizer import Tokenizer

if TYPE_CHECKING:
    from midigpt._types import Score
    from midigpt.inference.config import GenerationRequest
    from midigpt.inference.session import SamplingSession

_DEFAULT_HF_REPO = "Metacreation/MIDI-GPT"

_STEP_RE = re.compile(r"-step(\d+)\.safetensors$")

log = logging.getLogger(__name__)


def _resolve_checkpoint_filename(repo_id: str, prefix: str, token: str | None = None) -> str:
    """Pick the best checkpoint file in repo_id matching prefix.

    Prefers "<prefix>-final.safetensors"; otherwise picks the
    "<prefix>-step<N>.safetensors" file with the highest N.
    """
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(repo_id, token=token)
    candidates = [f for f in files if f.startswith(prefix + "-") and f.endswith(".safetensors")]
    if not candidates:
        raise ValueError(f"No checkpoint found in {repo_id!r} matching prefix {prefix!r}")
    return _pick_checkpoint(candidates, prefix)


def _pick_checkpoint(candidates, prefix: str) -> str:
    final = f"{prefix}-final.safetensors"
    if final in candidates:
        return final

    def step(fname: str) -> int:
        match = _STEP_RE.search(fname)
        return int(match.group(1)) if match else -1

    return max(candidates, key=step)


def _cached_checkpoint(repo_id: str, prefix: str) -> str | None:
    """Path of the checkpoint _resolve_checkpoint_filename would pick, from the
    local huggingface_hub cache only (the revision main was at when last
    downloaded, else the newest copy); None when nothing is cached."""
    from huggingface_hub.constants import HF_HUB_CACHE

    repo_dir = Path(HF_HUB_CACHE) / f"models--{repo_id.replace('/', '--')}"
    paths = [p for p in (repo_dir / "snapshots").glob(f"*/{prefix}-*.safetensors") if p.exists()]
    if not paths:
        return None
    fname = _pick_checkpoint({p.name for p in paths}, prefix)
    paths = [p for p in paths if p.name == fname]
    main = repo_dir / "refs" / "main"
    main_rev = main.read_text().strip() if main.is_file() else None
    return str(next((p for p in paths if p.parent.name == main_rev),
                    max(paths, key=lambda p: p.stat().st_mtime)))


def _is_gated(exc: Exception) -> bool:
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError

    if isinstance(exc, GatedRepoError):
        return True
    status = getattr(getattr(exc, "response", None), "status_code", None)
    return isinstance(exc, HfHubHTTPError) and status in (401, 403)


class InferenceEngine:
    def __init__(self, model, tokenizer: Tokenizer, analyzer: AttributeAnalyzer):
        self._model = model
        self._tokenizer = tokenizer
        self._analyzer = analyzer
        self._initial_kv = None  # cached once by warmup() or first generate

    @classmethod
    def from_pretrained(
        cls,
        name: str,
        hf_repo: str = _DEFAULT_HF_REPO,
        analyzer: AttributeAnalyzer | None = None,
        device: str | None = None,
        token: str | None = None,
    ) -> InferenceEngine:
        """Load a model from HuggingFace Hub.

        ``name`` is the checkpoint filename prefix on the repo, e.g.
        ``yellow_medium``, ``prism_medium``, ``expressive_medium``.
        The actual filename is resolved dynamically — prefers
        ``<name>-final.safetensors``, falls back to the highest-step snapshot.

        ``hf_repo`` defaults to ``Metacreation/MIDI-GPT``. Pass a custom repo
        ID to load from a different HuggingFace repository.

        The file is downloaded once and cached by ``huggingface_hub`` in
        ``~/.cache/huggingface/hub/``. When the Hub can't be reached, or
        refuses the download, a cached copy is used instead.

        ``token`` is a Hugging Face access token, needed for a gated repo
        once its license has been accepted on the Hub. When None, the token
        from ``HF_TOKEN`` or ``hf auth login`` is used.

        Examples::

            engine = InferenceEngine.from_pretrained("yellow_medium")
            engine = InferenceEngine.from_pretrained("prism_medium")
            engine = InferenceEngine.from_pretrained("my_model", hf_repo="myorg/myrepo")
        """
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise ImportError("pip install midigpt[inference] to enable HF Hub downloads") from None

        try:
            fname = _resolve_checkpoint_filename(hf_repo, name, token=token)
            local_path = hf_hub_download(repo_id=hf_repo, filename=fname, token=token)
        except Exception as exc:
            local_path = _cached_checkpoint(hf_repo, name)
            if local_path is None:
                if _is_gated(exc):
                    raise PermissionError(
                        f"{hf_repo} is gated. Accept its license at https://huggingface.co/{hf_repo} "
                        "with your Hugging Face account, then run `hf auth login`, set HF_TOKEN, "
                        "or pass token=..."
                    ) from exc
                raise
            log.warning("Could not download %s from %s (%s); using the cached %s",
                        name, hf_repo, type(exc).__name__, local_path)
        return cls.from_checkpoint(local_path, analyzer=analyzer, device=device)

    @classmethod
    def from_checkpoint(
        cls, path: str, analyzer: AttributeAnalyzer | None = None, device: str | None = None
    ) -> InferenceEngine:
        try:
            import torch
        except ImportError:
            raise ImportError("pip install midigpt[inference]") from None
        from midigpt.inference.model.torchscript_adapter import TorchScriptAdapter
        from midigpt.tokenizer.checkpoint import load_checkpoint

        bundle = load_checkpoint(path, device=device)
        if bundle.model is not None:
            model = bundle.model
        else:
            scripted = torch.jit.load(bundle.model_path, map_location=device or "cpu")
            scripted.eval()
            model = TorchScriptAdapter(scripted)
        tokenizer = Tokenizer(bundle.encoder_config, analyzer)
        engine = cls(
            model, tokenizer, analyzer or AttributeAnalyzer.from_config(bundle.encoder_config)
        )
        engine.warmup()
        return engine

    def warmup(self) -> None:
        """Disable TorchScript profiling and pre-build the empty KV cache."""
        import torch

        torch._C._jit_set_profiling_mode(False)  # type: ignore[attr-defined]
        torch._C._jit_set_profiling_executor(False)  # type: ignore[attr-defined]
        self._initial_kv = self._compute_initial_kv()

    def _compute_initial_kv(self):
        """Build an empty past_kv for the first model call.

        Requires ``model.make_empty_kv()`` (ModelBase protocol). Legacy
        TorchScript modules are wrapped in TorchScriptAdapter before reaching
        here, so every model has the method.
        """
        import torch

        model = self._model
        kv = model.make_empty_kv()
        # Derive device from the KV tensors; avoids StopIteration on stub models
        # that satisfy ModelBase without having nn.Module parameters.
        dev = torch.device("cpu")
        if kv and isinstance(kv[0][0], torch.Tensor):
            dev = kv[0][0].device
        with torch.no_grad():
            model(torch.tensor([[0]], dtype=torch.long, device=dev), kv)
        return kv

    def session(self, score: Score, request: GenerationRequest) -> SamplingSession:
        from midigpt.inference.session import SamplingSession
        from midigpt.inference.validation import validate_request

        request = validate_request(request, score, self._tokenizer._vocab.config(), self._analyzer)
        if self._initial_kv is None:
            self._initial_kv = self._compute_initial_kv()
        return SamplingSession(self, score, request)
