import midigpt._core as _core
from midigpt.tokenizer.tokenizer import Tokenizer
from midigpt.attributes.base import AttributeAnalyzer

HF_REPO_ID = "Metacreation/MIDI-GPT"

_KNOWN_MODELS: dict[str, str] = {
    "yellow":     "yellow.pt",
    "ghost":      "ghost.pt",
    "expressive": "expressive.pt",
}


class InferenceEngine:
    def __init__(self, model, tokenizer: Tokenizer, analyzer: AttributeAnalyzer):
        self._model     = model
        self._tokenizer = tokenizer
        self._analyzer  = analyzer
        self._initial_kv = None   # cached once by warmup() or first generate

    @classmethod
    def from_pretrained(
        cls,
        name_or_repo_id: str,
        filename: str | None = None,
        analyzer: "AttributeAnalyzer | None" = None,
    ) -> "InferenceEngine":
        """Load a model by name or HuggingFace repo ID.

        Short names resolve to files in the official repo::

            engine = InferenceEngine.from_pretrained("yellow")
            engine = InferenceEngine.from_pretrained("ghost")

        A full repo ID with an explicit filename also works::

            engine = InferenceEngine.from_pretrained(
                "Metacreation-Lab/MIDI-GPT", filename="yellow.pt"
            )

        The file is downloaded once and cached by ``huggingface_hub`` in
        ``~/.cache/huggingface/hub/``.
        """
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise ImportError("pip install midigpt[inference] to enable HF Hub downloads")

        if name_or_repo_id in _KNOWN_MODELS:
            repo_id = HF_REPO_ID
            fname   = filename or _KNOWN_MODELS[name_or_repo_id]
        else:
            if filename is None:
                raise ValueError(
                    f"Unknown model name {name_or_repo_id!r}. "
                    f"Known names: {list(_KNOWN_MODELS)}. "
                    f"For a custom repo pass filename= explicitly."
                )
            repo_id = name_or_repo_id
            fname   = filename

        local_path = hf_hub_download(repo_id=repo_id, filename=fname)
        return cls.from_checkpoint(local_path, analyzer=analyzer)

    @classmethod
    def from_checkpoint(cls, path: str,
                        analyzer: "AttributeAnalyzer | None" = None) -> "InferenceEngine":
        try:
            import torch
        except ImportError:
            raise ImportError("pip install midigpt[inference]")
        from midigpt.tokenizer.checkpoint import load_checkpoint
        from midigpt.inference.model.torchscript_adapter import TorchScriptAdapter
        bundle = load_checkpoint(path)
        if bundle.model is not None:
            model = bundle.model
        else:
            scripted = torch.jit.load(bundle.model_path, map_location="cpu")
            scripted.eval()
            model = TorchScriptAdapter(scripted)
        tokenizer = Tokenizer(bundle.encoder_config, analyzer)
        engine    = cls(model, tokenizer,
                        analyzer or AttributeAnalyzer.from_config(bundle.encoder_config))
        engine.warmup()
        return engine

    def warmup(self) -> None:
        """Disable TorchScript profiling and pre-build the empty KV cache."""
        import torch
        torch._C._jit_set_profiling_mode(False)        # type: ignore[attr-defined]
        torch._C._jit_set_profiling_executor(False)    # type: ignore[attr-defined]
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
        with torch.no_grad():
            model(torch.tensor([[0]], dtype=torch.long), kv)
        return kv

    def session(self, score: "Score",
                request: "GenerationRequest") -> "SamplingSession":
        from midigpt.inference.session import SamplingSession
        from midigpt.inference.validation import validate_request
        request = validate_request(
            request, score, self._tokenizer._vocab.config(), self._analyzer
        )
        if self._initial_kv is None:
            self._initial_kv = self._compute_initial_kv()
        return SamplingSession(self, score, request)
