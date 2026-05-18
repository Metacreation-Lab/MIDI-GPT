# orig C++ JIT vs refactor + SDPA — comparison benchmark

End-to-end full generation, identical inputs and sampling parameters
(`barsPerStep=2, modelDim=4, temp=1.0, max_attempts=1`).

- **orig** — `midigpt.sample_multi_step` (C++ TorchScript JIT, loads model on each call)
  - checkpoint: `models/yellow_orig_restored.pt` (reverse-remapped to orig vocab + injected `extra/metadata.json`)
- **ref** — `midigpt_refactor.InferenceEngine` with `GPT2LMHeadModel` (PyTorch + `F.scaled_dot_product_attention`, pre-loaded with KV cache)
  - checkpoint: `models/yellow.pt` (packed dict: `format_version=1`, embedded `config` + `encoder_config` + `state_dict`)
  - device: CPU (MPS/CUDA selectable via `MIDIGPT_BENCH_DEVICE`)

Ref one-time setup: **model_load=222.6ms · warmup=62.5ms** (excluded from per-call timings; for end-to-end add ~285ms to `r.med`).

## Results

| file | o.wall | r.med | r.min | speedup |
|---|---:|---:|---:|---:|
| 6338816_Etude No. 4.mid | 7001.8ms | 4042.4ms | 2058.4ms | **1.73x** |
| 6354774_Macabre Waltz.mid | 1403.3ms | 644.6ms | 321.7ms | **2.18x** |
| Aicha.mid | 2662.5ms | 644.4ms | 642.0ms | **4.13x** |
| All The Small Things.mid | 2269.2ms | 690.3ms | 686.6ms | **3.29x** |
| Funkytown.mid | 3569.7ms | 1479.0ms | 1464.6ms | **2.41x** |

Median speedup: **~2.4×** (median r-min speedup is higher — best-case ref is up to 4×).

## Breakdown (sum across steps)

| file | o.load | o.fwd | r.fwd |
|---|---:|---:|---:|
| Etude No. 4 | 160.3ms | 4626.2ms | 3824.8ms |
| Macabre Waltz | 154.2ms | 835.1ms | 598.2ms |
| Aicha | 197.9ms | 1596.1ms | 599.5ms |
| All The Small Things | 157.4ms | 1359.4ms | 643.2ms |
| Funkytown | 158.6ms | 2263.5ms | 1344.1ms |

- **Forward pass alone**: ref is consistently faster than orig (1.2×–2.7× on `fwd`), even though the per-call timing for orig still includes the ~155ms model load each time. SDPA + KV cache is the dominant win on `fwd`.
- **Tokenize/decode** are negligible on both paths.

## Notes

- Token/note counts differ between orig and ref on some files (e.g. Etude: 219 vs 131 toks). Same encoder config, same sampling params — divergence is expected because vocab-permutation + sampling stochasticity decorrelates the trajectories step-by-step. Quality parity is a separate test (`test_parity_raw.py`).
- The packed checkpoint format ships weights + arch config + encoder config in one file, so the refactor side needs no sidecar JSON.
- Device support: `resolve_device("auto")` picks CUDA > MPS > CPU. CPU is shown here; MPS path is available on Apple Silicon.

## Reproduce

```bash
.venv/bin/pytest midigpt_refactor/tests/comparison/test_benchmark.py::test_full_generation_speed -m benchmark -s

# device override:
MIDIGPT_BENCH_DEVICE=mps .venv/bin/pytest ... -m benchmark -s
```
