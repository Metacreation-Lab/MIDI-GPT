import midigpt._core as _core
from midigpt.tokenizer.tokenizer import Tokenizer
from midigpt.attributes.base import AttributeAnalyzer


class InferenceEngine:
    def __init__(self, model, tokenizer: Tokenizer, analyzer: AttributeAnalyzer):
        self._model     = model
        self._tokenizer = tokenizer
        self._analyzer  = analyzer
        self._initial_kv = None   # cached once by warmup() or first generate

    @classmethod
    def from_checkpoint(cls, path: str,
                        analyzer: AttributeAnalyzer | None = None) -> "InferenceEngine":
        try:
            import torch
        except ImportError:
            raise ImportError("pip install midigpt[inference]")
        from midigpt.tokenizer.checkpoint import load_checkpoint
        bundle    = load_checkpoint(path)
        if bundle.model is not None:
            model = bundle.model
        else:
            model = torch.jit.load(bundle.model_path, map_location="cpu")
            model.eval()
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

        Prefers model.make_empty_kv() when the model satisfies the ModelBase
        protocol. Falls back to a probe loop for TorchScript or other callables.
        """
        import torch
        model = self._model

        # Fast path: model implements ModelBase
        if hasattr(model, "make_empty_kv"):
            kv = model.make_empty_kv()
            with torch.no_grad():
                model(torch.tensor([[0]], dtype=torch.long), kv)
            return kv

        # Slow path: TorchScript or opaque callable — probe n_head
        try:
            trf    = model.transformer
            n_embd = trf.wte.weight.shape[1]
            n_layer = sum(1 for _ in trf.h.children())
        except Exception:
            return None

        for n_head in (8, 16, 12, 4):
            if n_embd % n_head != 0:
                continue
            head_dim = n_embd // n_head
            kv = tuple(
                (torch.zeros(1, n_head, 0, head_dim),
                 torch.zeros(1, n_head, 0, head_dim))
                for _ in range(n_layer)
            )
            try:
                with torch.no_grad():
                    model(torch.tensor([[0]], dtype=torch.long), kv)
                return kv
            except Exception:
                continue
        return None

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
