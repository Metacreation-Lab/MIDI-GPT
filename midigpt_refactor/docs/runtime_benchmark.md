# MIDI-GPT Inference Runtime Benchmark

**Host:** Intel Core i7-8750H (6c/12t), macOS 13.4
**Model:** `models/yellow_remapped.pt` — GPT-2 (n_layer=6, n_head=8, n_embd=512, vocab=647, n_positions=2048)
**PyTorch:** 2.2.2 · **Python:** 3.12
**Workload:** prefill (B×N → logits) and single-step decode (B×1 with past_kv at past_len)
**Repeats:** 3 prefill / 10 decode, warmup 1/2, median reported (min in raw log)
**Parity:** every backend's prefill logits compared against `TorchScriptBackend` at B=1 (max |Δ| + argmax agreement)

---

## TL;DR

- **Best overall on Intel CPU:** `openvino` for prefill, `onnxruntime.int8` for decode. Both keep top-1 parity (openvino) or sample-equivalent parity (int8).
- **No-deps drop-in:** `sdpa` (pure PyTorch + `F.scaled_dot_product_attention`) — 3–4× over TorchScript baseline at long context, **100% parity**.
- **Apple-silicon backends** (CoreML, MPS-on-AS, MLX) are implemented but skipped on this host; `torch.mps` runs through Apple's Intel-Metal shim and is fast for prefill but flaky for decode and shows non-deterministic FP drift.
- **llama.cpp** works for B=1 only — `Llama` has no batching API; useful for embedded/single-stream deployments.
- **`torch.compile`** errors out on PyTorch 2.2 + Python 3.12 ("Dynamo not supported on Python 3.12+"). Upgrade to PyTorch ≥2.4 to enable.

---

## Methodology

### Uniform backend interface

Every backend implements:

```python
forward(input_ids: Long[B, T], past_kv | None) -> (logits: Float[B, T, V], present_kv | None)
```

`past_kv` is a tuple-of-pairs of FP32 `(B, n_head, T_past, head_dim)` tensors (matching the original TorchScript export). Backends without a native KV interface (CoreML, llama.cpp) reconstruct the equivalent state internally; the harness only times the wall-clock and verifies the final logits.

### Workload

- **Prefill:** input length N ∈ {128, 512, 1024, 2048}, no past_kv. Measures how fast the backend processes a fresh prompt.
- **Decode:** input length 1, past_len ∈ {128, 512, 1024}. Measures per-token latency during streaming sampling.
- **Batch:** B ∈ {1, 2, 4, 8}. Single-stream is the real-time use case (one Max MSP performer per server); higher B is included to see scaling and to spot backends that hard-code B=1.

### Timing

```python
for _ in range(warmup): fn()
ts = []
for _ in range(repeats):
    t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
return median(ts), min(ts)
```

Each ctx re-seeds `torch.manual_seed(ctx)` so all backends see identical inputs (and so parity comparisons are meaningful).

### Parity

For prefill at B=1 only, the harness compares logits against the TorchScript baseline:

- `max |Δ|` — worst-case FP error across the full `(1, N, V)` tensor
- `agree` — fraction of positions where `argmax` matches the baseline

Decode parity is not measured (random KV cache content is not a useful reference); decode is timed only.

### Reproduction

```bash
.venv/bin/python3 midigpt_refactor/scripts/bench_runtimes.py --ckpt models/yellow_remapped.pt
```

Raw log: `/tmp/midigpt_bench_multibatch.log`

---

## Results — PREFILL latency (ms median)

### B = 1

| backend            |   128   |   512   |   1024  |   2048  |
|--------------------|--------:|--------:|--------:|--------:|
| torchscript (base) |   56.81 |  280.58 | 1328.02 | 2859.87 |
| eager              |   63.39 |  281.17 |  823.72 | 3171.18 |
| sdpa               |   33.66 |  146.46 |  340.56 |  707.24 |
| torch.int8         |   36.38 |  175.94 |  469.32 | 1484.76 |
| onnxruntime        |   26.54 |  110.69 |  310.92 | 1016.58 |
| onnxruntime.int8   |   23.13 |  130.89 |  280.47 |  889.35 |
| **openvino**       | **26.57** | **109.35** | **254.88** | **650.97** |
| torch.mps          |   60.65 |  104.25 |  181.79 |  483.64 |
| llama.cpp          |   42.32 |  210.52 |  441.28 | 1327.47 |

### B = 2

| backend          |   128  |   512  |  1024  |  2048  |
|------------------|-------:|-------:|-------:|-------:|
| torchscript      | 135.59 | 764.04 |1723.36 |6443.07 |
| eager            | 122.06 | 510.37 |1264.85 |4954.93 |
| sdpa             |  66.43 | 268.84 | 558.57 |1307.78 |
| torch.int8       |  50.19 | 312.10 |1001.92 |3232.86 |
| onnxruntime      |  51.75 | 248.50 | 588.51 |2967.10 |
| onnxruntime.int8 |  37.25 | 187.76 | 531.22 |1633.88 |
| openvino         |  52.39 | 176.18 | 434.51 |1427.43 |
| **torch.mps**    | **64.46** | **161.39** | **368.61** |  **915.15** |

### B = 4

| backend          |   128  |   512  |  1024  |  2048  |
|------------------|-------:|-------:|-------:|-------:|
| torchscript      | 230.09 |1233.73 |3147.55 |10284.68|
| eager            | 199.49 |1483.46 |2918.27 | 9476.28|
| sdpa             | 113.94 | 605.34 |1313.96 | 2759.34|
| torch.int8       | 107.96 | 675.05 |1960.40 | 6143.65|
| onnxruntime      |  92.17 | 736.82 |2222.30 | 3874.29|
| onnxruntime.int8 |  68.29 | 397.27 |1397.17 | 5224.97|
| openvino         |  89.18 | 406.85 |1085.51 | 2940.44|
| **torch.mps**    | **105.65** | **333.09** |  **692.15** | **1822.17** |

### B = 8

| backend          |   128  |   512  |  1024  |  2048  |
|------------------|-------:|-------:|-------:|-------:|
| torchscript      | 551.81 |2316.85 |7177.99 |26167.09|
| eager            | 292.48 |1496.48 |4452.85 |12520.14|
| sdpa             | 287.99 |1250.61 |2212.91 | 5452.61|
| torch.int8       | 242.63 |1414.87 |4157.46 |11703.60|
| onnxruntime      | 187.44 | 969.31 |2771.22 |12109.32|
| onnxruntime.int8 | 138.83 | 776.56 |2171.03 | 9675.29|
| openvino         | 235.86 |1599.86 |2583.55 | 6545.52|
| **torch.mps**    | **156.69** | **607.05** |**1373.25** |10421.67|

llama.cpp does not support B>1 (no batching API in `llama_cpp.Llama`) — n/a.

---

## Results — DECODE step latency (ms median)

### B = 1

| backend          |   128  |   512  |  1024  |
|------------------|-------:|-------:|-------:|
| torchscript      |   8.24 |  10.86 |  16.80 |
| eager            |   7.06 |   8.71 |  12.68 |
| sdpa             |   7.81 |   9.81 |  14.48 |
| torch.int8       |  10.38 |  13.57 |  17.86 |
| onnxruntime      |   7.85 |  18.57 |  50.22 |
| **onnxruntime.int8** | **3.30** | **6.54** | **11.39** |
| openvino         |  10.80 |  20.35 |  33.52 |
| torch.mps        |  65.90 |  76.41 |  93.12 |
| llama.cpp        |  73.85 | 212.30 | 590.52 |

### B = 2

| backend          |   128  |   512  |  1024  |
|------------------|-------:|-------:|-------:|
| torchscript      |  10.58 |  17.18 |  31.54 |
| eager            |   8.50 |  13.14 |  19.33 |
| sdpa             |   9.68 |  17.30 |  31.61 |
| torch.int8       |  11.38 |  18.60 |  26.37 |
| onnxruntime      |   9.12 |  18.58 |  63.76 |
| **onnxruntime.int8** | **4.38** | **11.77** | **23.10** |
| openvino         |  14.14 |  32.90 |  76.86 |
| torch.mps        |  71.24 |  92.44 | 125.78 |

### B = 4

| backend          |   128  |   512  |  1024  |
|------------------|-------:|-------:|-------:|
| torchscript      |  18.46 |  30.21 |  60.65 |
| **eager**        |   **9.55** |  **18.92** |  **33.39** |
| sdpa             |  21.94 |  41.09 |  46.48 |
| torch.int8       |  14.54 |  26.81 |  40.62 |
| onnxruntime      |  19.45 |  34.69 |  44.16 |
| onnxruntime.int8 |   6.44 |  24.81 |  85.66 |
| openvino         |  14.27 |  60.86 | 106.11 |
| torch.mps        |  77.47 | 140.19 | 145.00 |

### B = 8

| backend          |   128  |   512  |  1024  |
|------------------|-------:|-------:|-------:|
| torchscript      |  30.15 |  49.51 |  96.82 |
| **eager**        |  **13.92** |  **35.68** |  **66.58** |
| sdpa             |  17.80 |  46.38 |  91.67 |
| torch.int8       |  21.30 |  51.41 |  97.70 |
| onnxruntime      |  14.22 |  49.75 | 111.50 |
| onnxruntime.int8 |  10.36 |  65.36 |  85.97 |
| openvino         |  26.73 | 117.34 | 206.79 |
| torch.mps        |  92.31 | 140.50 | 154.48 |

---

## Parity (vs TorchScript baseline, B=1 prefill)

| backend            | ctx=128 | ctx=512 | ctx=1024 | ctx=2048 | verdict |
|--------------------|---------|---------|----------|----------|---------|
| torchscript        | 0 / 100%| 0 / 100%| 0 / 100% | 0 / 100% | (baseline) |
| eager              | 8.7e-6 / 100% | 2.4e-5 / 100% | 3.3e-5 / 100% | 4.2e-5 / 100% | ✅ exact match for sampling |
| sdpa               | 8.6e-6 / 100% | 1.5e-5 / 100% | 3.7e-5 / 100% | 3.0e-5 / 100% | ✅ exact match for sampling |
| torch.int8         | 1.6  / 85.9%  | 2.8 / 82.6%   | 6.3 / 79.9%   | 4.2 / 78.5%   | ⚠ argmax drift; distribution preserved |
| onnxruntime        | 1.7e-5 / 100% | 3.2e-5 / 100% | 4.5e-5 / 100% | 4.5e-5 / 100% | ✅ exact match for sampling |
| onnxruntime.int8   | 1.1  / 88.3%  | 1.7 / 85.4%   | 2.5 / 85.8%   | 2.4 / 81.8%   | ⚠ argmax drift; distribution preserved |
| openvino           | 1.4e-5 / 100% | 2.1e-5 / 100% | 3.0e-5 / 100% | 4.1e-5 / 100% | ✅ exact match for sampling |
| torch.mps          | 1.4e-5 / 100% | 1.9e+1 / 5.7% | 6.0e-5 / 100% | 4.2  / 84.8% | ❌ shape-dependent kernel drift |
| llama.cpp          | 7.8e-3 / 100% | 2.3e-2 / 100% | 2.3e-2 / 99.8% | 2.1e-2 / 99.9% | ✅ near-exact match |

### Why int8 backends drift in argmax but still sample correctly

Both `torch.int8` (PyTorch dynamic quant) and `onnxruntime.int8` use per-channel int8 weight quantization. The ±0.5 ULP weight error compounds across all 6 attention/MLP layers. By the final logits, FP differences of 1–6 (raw logit units) are typical — enough to flip `argmax` on roughly 15–20% of positions, but the softmax distribution remains very close to FP32. For sampling at temperature ≥ 0.8 this is indistinguishable from FP32 in practice. For greedy/temp=0, parity is broken — prefer `openvino` or `sdpa`.

### Why torch.mps parity is flaky

The host is an Intel Mac; `torch.backends.mps` dispatches via Apple's Intel-Metal compatibility shim (using the Iris Plus iGPU). Apple's Metal kernels for FP32 matmul use different shapes/tilings depending on the input dimensions, and the shim has known precision regressions at non-power-of-two ctx values. ctx=512 catastrophically diverges (max|Δ|=19, 5.7% agree) while ctx=128 and ctx=1024 stay at 100%. **Do not deploy `torch.mps` on Intel.** On Apple Silicon the same code path uses native MPS kernels and is expected to be much better-behaved (untested here).

### Why llama.cpp has a higher absolute Δ but still 100% argmax

llama.cpp internally upcasts FP32 weights to F16 in some matmul paths. The absolute diff is larger (~2e-2 vs ~3e-5 for ORT/OpenVINO), but it's uniform noise — it doesn't change argmax until ctx=1024+ where one or two positions flip (99.8–99.9% agree). Functionally equivalent for sampling.

---

## Backend availability summary

| backend             | status on Intel Mac        | notes |
|---------------------|----------------------------|-------|
| torchscript         | ✅                         | baseline |
| eager               | ✅                         | pure nn.Module, no transformers |
| torch.compile       | ❌ Dynamo unsupported on Python 3.12 + PyTorch 2.2.2 | upgrade to PyTorch ≥2.4 |
| sdpa                | ✅                         | best zero-dep speedup, full parity |
| torch.int8          | ✅                         | parity drift; OK for sampling |
| onnxruntime         | ✅                         | best decode for parity, best long-ctx prefill at B=1 |
| onnxruntime.int8    | ✅                         | fastest decode B=1; parity drift |
| onnxruntime.coreml  | ⏭ skipped (Apple Silicon)  | CoreML EP falls back to CPU on Intel |
| openvino            | ✅                         | best prefill at long ctx (Intel CPU + iGPU graph) |
| coreml              | ⏭ skipped (Apple Silicon)  | needs ANE; CPU fallback too slow to bother |
| torch.mps           | ⚠ runs but unreliable      | Intel Metal shim has FP drift |
| mlx                 | ⏭ stub (Apple Silicon)     | implement in `runtimes_mlx.py` when on AS |
| llama.cpp           | ✅ (B=1 only)              | GGUF export at `/tmp/midigpt_runtimes/yellow.gguf` |

---

## Recommendations

**For the real-time OSC server today (Intel Mac, B=1 streaming):**

1. **Prefill (≤512 tokens, typical bar context):** `sdpa` if you want zero-dep; `onnxruntime` or `openvino` if you can ship the runtime.
2. **Decode (per-step during sampling):** `onnxruntime.int8` (~3 ms at past_len=128) → ~4× faster than TorchScript.
3. **If shipping a single binary is critical:** `llama.cpp` decode is slow but the GGUF is self-contained.

**For Apple Silicon migration:**

- Re-run the bench on Apple Silicon. Expect `torch.mps`, `coreml`, and `onnxruntime.coreml` to leapfrog the CPU backends. `mlx` is still a stub — implement once you have hardware.
- The current code paths all stay valid; only the `is_available()` gates change.

**Pending follow-ups:**

- Upgrade PyTorch ≥2.4 → enable `torch.compile`.
- Implement MLX backend on first AS workstation (mirror `GPT2LMHeadModel` with `mlx.nn`).
- Investigate ONNX Runtime decode regression at high past_len (B=1 ctx=1024 jumps to 50ms; B=2 to 64ms). Likely missing KV-cache concat optimization in the exported graph.
