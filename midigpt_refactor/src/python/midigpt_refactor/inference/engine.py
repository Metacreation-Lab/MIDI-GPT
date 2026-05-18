import midigpt_refactor._core as _core
from midigpt_refactor.tokenizer.tokenizer import Tokenizer
from midigpt_refactor.attributes.base import AttributeAnalyzer


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
        from midigpt_refactor.tokenizer.checkpoint import load_checkpoint
        bundle    = load_checkpoint(path)
        model     = torch.jit.load(bundle.model_path, map_location="cpu")
        model.eval()
        tokenizer = Tokenizer(bundle.encoder_config, analyzer)
        engine    = cls(model, tokenizer,
                        analyzer or AttributeAnalyzer.from_config(bundle.encoder_config))
        engine.warmup()
        return engine

    def warmup(self) -> None:
        """Disable TorchScript's profiling executor (which causes shape-specific
        re-optimization spikes and adds per-call overhead) and probe the KV
        cache shape.  After this, the first real generate() runs at
        steady-state speed.
        """
        import torch
        torch._C._jit_set_profiling_mode(False)        # type: ignore[attr-defined]
        torch._C._jit_set_profiling_executor(False)    # type: ignore[attr-defined]
        self._initial_kv = self._compute_initial_kv()

    def _compute_initial_kv(self):
        """Build empty past_key_values matching the model's expected signature.

        GPT-2 TorchScript export shape: (batch=1, n_heads, seq=0, head_dim).
        Tries common n_head values; returns None if the model doesn't accept KV.
        """
        import torch
        try:
            trf    = self._model.transformer
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
                    # also triggers JIT compilation for this input shape
                    self._model(torch.tensor([[0]], dtype=torch.long), kv)
                return kv
            except Exception:
                continue
        return None

    def session(self, score: "Score",
                request: "GenerationRequest") -> "SamplingSession":
        from midigpt_refactor.inference.session import SamplingSession
        from midigpt_refactor.inference.validation import validate_request
        request = validate_request(
            request, score, self._tokenizer._vocab.config(), self._analyzer
        )
        # lazily compute KV on first generate if warmup() was not called
        if self._initial_kv is None:
            self._initial_kv = self._compute_initial_kv()
        return SamplingSession(self, score, request)
