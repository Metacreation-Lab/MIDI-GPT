"""Pluggable inference backends for the GPT-2 model.

Each backend wraps a forward function with a uniform signature:

    forward(input_ids: LongTensor[B, T], past_kv | None) ->
        (logits: FloatTensor[B, T, V], present_kv)

`past_kv` is a tuple-of-pairs of FP32 tensors shaped `(B, n_head, T_past, head_dim)`
(matching the original TorchScript export).  `present_kv` is the same shape with
`T_past + T` at the seq dimension.  Backends that don't natively return KV
caches return `None`; the bench harness handles that.

The harness uses `is_available()` to skip backends whose deps are missing or
that target a different platform than the host.

Supported backends:
    1. TorchScriptBackend             — current production
    2. EagerBackend                   — pure nn.Module
    3. CompileBackend                 — torch.compile (TorchInductor)
    4. SDPABackend                    — eager + torch.scaled_dot_product_attention
    5. DynInt8Backend                 — torch.quantization.quantize_dynamic
    6. ONNXRuntimeBackend             — ORT CPU
    7. ONNXRuntimeInt8Backend         — ORT CPU + int8 dynamic quant
    8. ONNXRuntimeCoreMLBackend       — ORT + CoreMLExecutionProvider (Apple)
    9. OpenVINOBackend                — Intel OpenVINO
   10. CoreMLBackend                  — .mlpackage via coremltools (Apple)
   11. MPSBackend                     — PyTorch MPS device (Apple)
   12. MLXBackend                     — Apple MLX framework
   13. LlamaCppBackend                — GGUF + llama.cpp
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
import math
import platform
import torch
import torch.nn as nn

from midigpt_refactor.inference.model import GPT2LMHeadModel, GPT2Config


# --------------------------------------------------------------------------- #
#  Base class
# --------------------------------------------------------------------------- #
class Backend(ABC):
    name: str = "abstract"

    @classmethod
    def is_available(cls) -> tuple[bool, str]:
        """Return (available, reason). Reason is shown when skipped."""
        return True, ""

    @abstractmethod
    def setup(self, ckpt_path: str) -> None: ...

    @abstractmethod
    def forward(self, input_ids, past_kv=None): ...


def _empty_past(cfg: GPT2Config, batch: int = 1):
    return tuple(
        (torch.zeros(batch, cfg.n_head, 0, cfg.head_dim),
         torch.zeros(batch, cfg.n_head, 0, cfg.head_dim))
        for _ in range(cfg.n_layer)
    )


# --------------------------------------------------------------------------- #
#  1. TorchScript (baseline)
# --------------------------------------------------------------------------- #
class TorchScriptBackend(Backend):
    name = "torchscript"

    def setup(self, ckpt_path):
        self.model = torch.jit.load(ckpt_path, map_location="cpu")
        self.model.eval()
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)
        # probe cfg
        sd = self.model.state_dict()
        D = sd["transformer.wte.weight"].shape[1]
        V = sd["transformer.wte.weight"].shape[0]
        P = sd["transformer.wpe.weight"].shape[0]
        nL = sum(1 for k in sd if k.endswith(".attn.c_attn.weight"))
        self.cfg = GPT2Config(vocab_size=V, n_positions=P, n_embd=D, n_layer=nL, n_head=8)

    def forward(self, input_ids, past_kv=None):
        if past_kv is None:
            past_kv = _empty_past(self.cfg, input_ids.shape[0])
        with torch.no_grad():
            out = self.model(input_ids, past_kv)
        # TorchScript yellow.pt returns logits only (Tensor); some exports
        # return (logits, presents). Normalize to (logits, None).
        logits = out[0] if isinstance(out, (tuple, list)) else out
        return logits, None


# --------------------------------------------------------------------------- #
#  2. PyTorch eager (pure nn.Module)
# --------------------------------------------------------------------------- #
class EagerBackend(Backend):
    name = "eager"

    def setup(self, ckpt_path):
        self.model = GPT2LMHeadModel.from_torchscript(ckpt_path)
        self.cfg = self.model.cfg

    def forward(self, input_ids, past_kv=None):
        with torch.no_grad():
            return self.model(input_ids, past_kv=past_kv)


# --------------------------------------------------------------------------- #
#  3. torch.compile (TorchInductor)
# --------------------------------------------------------------------------- #
class CompileBackend(Backend):
    name = "torch.compile"

    @classmethod
    def is_available(cls):
        try:
            import torch._dynamo  # noqa
            return True, ""
        except Exception as e:
            return False, f"torch._dynamo unavailable: {e}"

    def setup(self, ckpt_path):
        eager = GPT2LMHeadModel.from_torchscript(ckpt_path)
        self.cfg = eager.cfg
        self.model = torch.compile(eager, mode="reduce-overhead", fullgraph=False)

    def forward(self, input_ids, past_kv=None):
        with torch.no_grad():
            return self.model(input_ids, past_kv=past_kv)


# --------------------------------------------------------------------------- #
#  4. SDPA attention (drop-in replacement for manual softmax)
# --------------------------------------------------------------------------- #
class _GPT2AttentionSDPA(nn.Module):
    """Re-implements the attention block using torch.scaled_dot_product_attention."""
    def __init__(self, original):
        super().__init__()
        self.n_head = original.n_head
        self.head_dim = original.head_dim
        self.c_attn = original.c_attn
        self.c_proj = original.c_proj
        self.register_buffer("_dummy", torch.empty(0), persistent=False)

    def _split(self, x):
        B, T, _ = x.shape
        return x.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

    def _merge(self, x):
        B, _, T, _ = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, self.n_head * self.head_dim)

    def forward(self, x, past_kv=None):
        qkv = self.c_attn(x)
        q, k, v = qkv.split(qkv.shape[-1] // 3, dim=-1)
        q, k, v = self._split(q), self._split(k), self._split(v)

        if past_kv is not None and past_kv[0].shape[2] > 0:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)

        # SDPA: is_causal=True handles square q×k; for q<k (decode), we need to
        # disable is_causal and pass an explicit mask.
        T_q, T_k = q.shape[2], k.shape[2]
        if T_q == T_k:
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            # decode-step or prefill-with-cache: build position mask
            mask = torch.zeros(T_q, T_k, dtype=torch.bool, device=q.device)
            offset = T_k - T_q
            for i in range(T_q):
                mask[i, : offset + i + 1] = True
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        y = self._merge(y)
        return self.c_proj(y), (k, v)


class SDPABackend(Backend):
    name = "sdpa"

    def setup(self, ckpt_path):
        m = GPT2LMHeadModel.from_torchscript(ckpt_path)
        for blk in m.transformer.h:
            blk.attn = _GPT2AttentionSDPA(blk.attn)
        m.eval()
        self.model = m
        self.cfg = m.cfg

    def forward(self, input_ids, past_kv=None):
        with torch.no_grad():
            return self.model(input_ids, past_kv=past_kv)


# --------------------------------------------------------------------------- #
#  5. PyTorch dynamic int8 quantization
# --------------------------------------------------------------------------- #
class DynInt8Backend(Backend):
    name = "torch.int8"

    def setup(self, ckpt_path):
        m = GPT2LMHeadModel.from_torchscript(ckpt_path)
        # quantize linear modules only (Conv1D is a Linear-equivalent)
        # we have to convert Conv1D to Linear first since quantize_dynamic
        # only matches nn.Linear by default.
        from midigpt_refactor.inference.model import Conv1D
        for mod in m.modules():
            for name, child in list(mod.named_children()):
                if isinstance(child, Conv1D):
                    nx, nf = child.weight.shape
                    lin = nn.Linear(nx, nf, bias=True)
                    with torch.no_grad():
                        lin.weight.copy_(child.weight.t())
                        lin.bias.copy_(child.bias)
                    setattr(mod, name, lin)
        self.model = torch.quantization.quantize_dynamic(
            m, {nn.Linear}, dtype=torch.qint8
        )
        self.cfg = m.cfg

    def forward(self, input_ids, past_kv=None):
        with torch.no_grad():
            return self.model(input_ids, past_kv=past_kv)


# --------------------------------------------------------------------------- #
#  6. ONNX Runtime CPU
# --------------------------------------------------------------------------- #
class _ORTBase(Backend):
    """Shared base for ONNX Runtime backends."""
    providers: tuple[str, ...] = ("CPUExecutionProvider",)
    onnx_int8: bool = False

    @classmethod
    def is_available(cls):
        try:
            import onnxruntime  # noqa
            return True, ""
        except Exception as e:
            return False, f"onnxruntime not installed: {e}"

    def setup(self, ckpt_path):
        import onnxruntime as ort
        from pathlib import Path
        from midigpt_refactor.inference.export_onnx import export as _export

        out_dir = Path("/tmp/midigpt_runtimes")
        out_dir.mkdir(exist_ok=True)
        prefill = out_dir / "yellow_prefill.onnx"
        decode  = out_dir / "yellow_decode.onnx"
        if not prefill.exists() or not decode.exists():
            _export(ckpt_path, str(prefill), str(decode))

        if self.onnx_int8:
            try:
                from onnxruntime.quantization import quantize_dynamic, QuantType
                pq = out_dir / "yellow_prefill_int8.onnx"
                dq = out_dir / "yellow_decode_int8.onnx"
                if not pq.exists():
                    quantize_dynamic(str(prefill), str(pq), weight_type=QuantType.QInt8)
                if not dq.exists():
                    quantize_dynamic(str(decode), str(dq), weight_type=QuantType.QInt8)
                prefill, decode = pq, dq
            except Exception as e:
                raise RuntimeError(f"int8 quant failed: {e}")

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.prefill_sess = ort.InferenceSession(str(prefill), opts, providers=list(self.providers))
        self.decode_sess  = ort.InferenceSession(str(decode),  opts, providers=list(self.providers))

        eager = GPT2LMHeadModel.from_torchscript(ckpt_path)
        self.cfg = eager.cfg

    def forward(self, input_ids, past_kv=None):
        import numpy as np
        ids = input_ids.detach().cpu().numpy().astype("int64")
        use_decode = past_kv is not None and past_kv[0][0].shape[2] > 0
        if use_decode:
            feed = {"input_ids": ids}
            for i, (k, v) in enumerate(past_kv):
                feed[f"past_{i}_k"] = k.detach().cpu().numpy().astype("float32")
                feed[f"past_{i}_v"] = v.detach().cpu().numpy().astype("float32")
            outs = self.decode_sess.run(None, feed)
        else:
            outs = self.prefill_sess.run(None, {"input_ids": ids})
        logits = torch.from_numpy(outs[0])
        # rebuild present
        n = self.cfg.n_layer
        present = tuple(
            (torch.from_numpy(outs[1 + 2*i]), torch.from_numpy(outs[1 + 2*i + 1]))
            for i in range(n)
        )
        return logits, present


class ONNXRuntimeBackend(_ORTBase):
    name = "onnxruntime"
    providers = ("CPUExecutionProvider",)


class ONNXRuntimeInt8Backend(_ORTBase):
    name = "onnxruntime.int8"
    providers = ("CPUExecutionProvider",)
    onnx_int8 = True


class ONNXRuntimeCoreMLBackend(_ORTBase):
    """Apple-only. CoreMLExecutionProvider falls back to CPU for unsupported ops."""
    name = "onnxruntime.coreml"
    providers = ("CoreMLExecutionProvider", "CPUExecutionProvider")

    @classmethod
    def is_available(cls):
        ok, msg = _ORTBase.is_available()
        if not ok:
            return False, msg
        if platform.machine() == "x86_64":
            return False, "CoreML EP is Apple-Silicon-only (will fall back to CPU on Intel)"
        try:
            import onnxruntime as ort
            if "CoreMLExecutionProvider" not in ort.get_available_providers():
                return False, "CoreMLExecutionProvider not in onnxruntime build (Apple Silicon only)"
            return True, ""
        except Exception as e:
            return False, str(e)


# --------------------------------------------------------------------------- #
#  7. OpenVINO (Intel)
# --------------------------------------------------------------------------- #
class OpenVINOBackend(Backend):
    name = "openvino"

    @classmethod
    def is_available(cls):
        try:
            import openvino  # noqa
            return True, ""
        except Exception as e:
            return False, f"openvino not installed: {e}"

    def setup(self, ckpt_path):
        from pathlib import Path
        import openvino as ov
        from midigpt_refactor.inference.export_onnx import export as _export

        out_dir = Path("/tmp/midigpt_runtimes")
        out_dir.mkdir(exist_ok=True)
        prefill = out_dir / "yellow_prefill.onnx"
        decode  = out_dir / "yellow_decode.onnx"
        if not prefill.exists() or not decode.exists():
            _export(ckpt_path, str(prefill), str(decode))

        core = ov.Core()
        self.prefill = core.compile_model(core.read_model(str(prefill)), "CPU")
        self.decode  = core.compile_model(core.read_model(str(decode)),  "CPU")
        eager = GPT2LMHeadModel.from_torchscript(ckpt_path)
        self.cfg = eager.cfg

    def forward(self, input_ids, past_kv=None):
        import numpy as np
        ids = input_ids.detach().cpu().numpy().astype("int64")
        use_decode = past_kv is not None and past_kv[0][0].shape[2] > 0
        if use_decode:
            feed = {"input_ids": ids}
            for i, (k, v) in enumerate(past_kv):
                feed[f"past_{i}_k"] = k.detach().cpu().numpy().astype("float32")
                feed[f"past_{i}_v"] = v.detach().cpu().numpy().astype("float32")
            out = self.decode(feed)
        else:
            out = self.prefill({"input_ids": ids})

        # OpenVINO returns dict keyed by output name
        keys = sorted(out.keys(), key=lambda k: k.any_name)
        logits = torch.from_numpy(out[next(k for k in out if k.any_name == "logits")])
        n = self.cfg.n_layer
        present = tuple(
            (torch.from_numpy(out[next(k for k in out if k.any_name == f"present_{i}_k")]),
             torch.from_numpy(out[next(k for k in out if k.any_name == f"present_{i}_v")]))
            for i in range(n)
        )
        return logits, present


# --------------------------------------------------------------------------- #
#  8. CoreML (.mlpackage) — Apple-only
# --------------------------------------------------------------------------- #
class CoreMLBackend(Backend):
    name = "coreml"

    @classmethod
    def is_available(cls):
        try:
            import coremltools  # noqa
        except Exception as e:
            return False, f"coremltools not installed: {e}"
        if platform.system() != "Darwin":
            return False, "CoreML requires macOS"
        if platform.machine() == "x86_64":
            return False, "CoreML prediction only runs on Apple Silicon at full speed"
        return True, ""

    def setup(self, ckpt_path):
        import coremltools as ct
        from pathlib import Path
        eager = GPT2LMHeadModel.from_torchscript(ckpt_path)
        eager.eval()
        self.cfg = eager.cfg

        out_dir = Path("/tmp/midigpt_runtimes")
        out_dir.mkdir(exist_ok=True)
        ml_path = out_dir / "yellow_coreml.mlpackage"
        if not ml_path.exists():
            example_ids = torch.zeros(1, 16, dtype=torch.long)
            traced = torch.jit.trace(
                lambda x: eager(x)[0],
                (example_ids,),
                strict=False,
            )
            mlmodel = ct.convert(
                traced,
                inputs=[ct.TensorType(name="input_ids",
                                       shape=ct.Shape([1, ct.RangeDim(1, eager.cfg.n_positions)]),
                                       dtype=int)],
                minimum_deployment_target=ct.target.macOS13,
                compute_units=ct.ComputeUnit.ALL,
            )
            mlmodel.save(str(ml_path))
        self.model = ct.models.MLModel(str(ml_path))

    def forward(self, input_ids, past_kv=None):
        import numpy as np
        out = self.model.predict({"input_ids": input_ids.detach().cpu().numpy().astype(np.int32)})
        logits = torch.from_numpy(np.asarray(list(out.values())[0]))
        return logits, None


# --------------------------------------------------------------------------- #
#  9. PyTorch MPS — Apple-only
# --------------------------------------------------------------------------- #
class MPSBackend(Backend):
    name = "torch.mps"

    @classmethod
    def is_available(cls):
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            return False, "MPS not available (Apple Silicon + macOS 12.3+ required)"
        return True, ""

    def setup(self, ckpt_path):
        m = GPT2LMHeadModel.from_torchscript(ckpt_path).to("mps")
        self.model = m
        self.cfg = m.cfg

    def forward(self, input_ids, past_kv=None):
        input_ids = input_ids.to("mps")
        if past_kv is not None:
            past_kv = tuple((k.to("mps"), v.to("mps")) for k, v in past_kv)
        with torch.no_grad():
            logits, present = self.model(input_ids, past_kv=past_kv)
        return logits.cpu(), tuple((k.cpu(), v.cpu()) for k, v in present)


# --------------------------------------------------------------------------- #
#  10. MLX (Apple)
# --------------------------------------------------------------------------- #
class MLXBackend(Backend):
    name = "mlx"

    @classmethod
    def is_available(cls):
        try:
            import mlx.core  # noqa
            return True, ""
        except Exception as e:
            return False, f"mlx not installed (Apple Silicon only): {e}"

    def setup(self, ckpt_path):
        raise NotImplementedError(
            "MLX port is a stub. Implement in midigpt_refactor/inference/runtimes_mlx.py "
            "by mirroring GPT2LMHeadModel using mlx.nn primitives and loading the "
            "remapped state_dict via mlx.utils.tree_unflatten."
        )

    def forward(self, input_ids, past_kv=None):
        raise NotImplementedError


# --------------------------------------------------------------------------- #
#  11. llama.cpp / GGUF
# --------------------------------------------------------------------------- #
class LlamaCppBackend(Backend):
    name = "llama.cpp"

    @classmethod
    def is_available(cls):
        try:
            import llama_cpp  # noqa
            return True, ""
        except Exception as e:
            return False, (
                "llama-cpp-python not installed; install with "
                f"`pip install llama-cpp-python`. Detail: {e}"
            )

    def setup(self, ckpt_path):
        from pathlib import Path
        import llama_cpp
        gguf = Path("/tmp/midigpt_runtimes/yellow.gguf")
        if not gguf.exists():
            from midigpt_refactor.inference.export_gguf import export as _export
            _export(ckpt_path, str(gguf))
        self.llm = llama_cpp.Llama(
            model_path=str(gguf),
            n_ctx=2048, n_batch=512, n_threads=6, logits_all=True, verbose=False,
        )
        eager = GPT2LMHeadModel.from_torchscript(ckpt_path)
        self.cfg = eager.cfg
        self._gguf_path = str(gguf)

    def forward(self, input_ids, past_kv=None):
        # llama.cpp maintains internal KV state. To emulate the external past_kv
        # interface, we reset and either eval input_ids (prefill) or eval
        # past_len filler tokens then the input (decode-step). Logits are
        # taken from positions [n_tokens - T, n_tokens).
        import numpy as np
        if input_ids.shape[0] != 1:
            raise NotImplementedError("llama.cpp backend does not support B>1")
        ids = input_ids[0].tolist()
        past_len = 0
        if past_kv is not None and past_kv[0][0].shape[2] > 0:
            past_len = past_kv[0][0].shape[2]
        self.llm.reset()
        if past_len > 0:
            # Filler tokens to populate KV cache up to past_len
            self.llm.eval([0] * past_len)
        self.llm.eval(ids)
        T = len(ids)
        n = self.llm.n_tokens
        sc = np.array(self.llm.scores[n - T : n], dtype=np.float32)
        return torch.from_numpy(sc).unsqueeze(0), None


# --------------------------------------------------------------------------- #
#  Registry
# --------------------------------------------------------------------------- #
ALL_BACKENDS: list[type[Backend]] = [
    TorchScriptBackend,
    EagerBackend,
    CompileBackend,
    SDPABackend,
    DynInt8Backend,
    ONNXRuntimeBackend,
    ONNXRuntimeInt8Backend,
    ONNXRuntimeCoreMLBackend,
    OpenVINOBackend,
    CoreMLBackend,
    MPSBackend,
    MLXBackend,
    LlamaCppBackend,
]
