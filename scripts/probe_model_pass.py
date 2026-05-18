"""
Probe per-call latency of the raw TorchScript model.forward path.

Goal: isolate the "first-bar spike" from the rest of the inference pipeline
(tokenizer, analyzer, grammar, sampler).  Just load the model, run random
forward passes with varying lengths and varying batch shapes, and time each.

If a spike still appears here even after a synthetic warmup, the cost is
inside model.forward — most likely TorchScript's profiling executor running
re-optimization passes after observing a new shape.

Run:
    .venv/bin/python3 scripts/probe_model_pass.py --ckpt models/yellow.pt

Optional flags:
    --vocab-size 647      vocab cap for random ids
    --n-warmups 1         warmup forward passes (paid before timing starts)
    --warmup-len 1024     length of each warmup pass
    --n-iters 30          timed forward passes
    --shapes random|fixed shape regime for timed loop
    --disable-profiling   set torch._C._jit_set_profiling_{mode,executor}(False)
"""
from __future__ import annotations

import argparse
import random
import statistics
import time
from pathlib import Path
from typing import Optional

import torch


def _load_model(ckpt_path: str) -> torch.jit.ScriptModule:
    """Load the model — supports both bare TorchScript files and the
    bundled .pt that midigpt ships (load_checkpoint returns a bundle)."""
    # Try midigpt_refactor's loader first (handles the bundle), fall back to raw.
    try:
        from midigpt_refactor.tokenizer.checkpoint import load_checkpoint
        bundle = load_checkpoint(ckpt_path)
        return torch.jit.load(bundle.model_path, map_location="cpu")
    except Exception:
        pass
    return torch.jit.load(ckpt_path, map_location="cpu")


def _build_empty_kv(model: torch.jit.ScriptModule):
    """Probe the model architecture to build an empty past_key_values cache.
    Returns the kv tuple or None if the model doesn't accept KV."""
    try:
        trf     = model.transformer
        n_embd  = trf.wte.weight.shape[1]
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


def _forward(model, ids: torch.Tensor, kv) -> float:
    """Time a single forward pass.  Returns wall-clock seconds."""
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.perf_counter()
    with torch.no_grad():
        if kv is None:
            model(ids)
        else:
            model(ids, kv)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",      required=True)
    ap.add_argument("--vocab-size", type=int, default=647)
    ap.add_argument("--n-warmups",  type=int, default=1)
    ap.add_argument("--warmup-len", type=int, default=1024)
    ap.add_argument("--n-iters",    type=int, default=30)
    ap.add_argument("--shapes",     choices=["random", "fixed"], default="random")
    ap.add_argument("--fixed-len",  type=int, default=256)
    ap.add_argument("--min-len",    type=int, default=64)
    ap.add_argument("--max-len",    type=int, default=1024)
    ap.add_argument("--seed",       type=int, default=0)
    ap.add_argument("--disable-profiling", action="store_true",
                    help="Disable TorchScript profiling executor / mode.")
    args = ap.parse_args()

    if args.disable_profiling:
        torch._C._jit_set_profiling_mode(False)        # type: ignore[attr-defined]
        torch._C._jit_set_profiling_executor(False)    # type: ignore[attr-defined]
        print("[profiling executor & mode disabled]")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Loading {args.ckpt} …")
    t0 = time.perf_counter()
    model = _load_model(args.ckpt)
    model.eval()
    print(f"  loaded in {(time.perf_counter() - t0)*1000:.0f}ms")

    print("Probing KV cache shape …")
    kv = _build_empty_kv(model)
    print(f"  kv supported: {kv is not None}"
          f"{f' (n_layer={len(kv)}, n_head={kv[0][0].shape[1]}, head_dim={kv[0][0].shape[3]})' if kv else ''}")

    # ---------- WARMUP ----------
    print(f"\nWarmup: {args.n_warmups} forward pass(es) at len={args.warmup_len}")
    for i in range(args.n_warmups):
        ids = torch.randint(0, args.vocab_size, (1, args.warmup_len), dtype=torch.long)
        dt  = _forward(model, ids, kv)
        print(f"  warmup[{i}]  len={args.warmup_len:>4d}  {dt*1000:8.1f}ms")

    # ---------- TIMED LOOP ----------
    print(f"\nTimed loop: {args.n_iters} passes, shapes={args.shapes}")
    times: list[float] = []
    lengths: list[int] = []
    for i in range(args.n_iters):
        if args.shapes == "random":
            L = random.randint(args.min_len, args.max_len)
        else:
            L = args.fixed_len
        ids = torch.randint(0, args.vocab_size, (1, L), dtype=torch.long)
        dt  = _forward(model, ids, kv)
        times.append(dt)
        lengths.append(L)
        marker = "  ← FIRST"  if i == 0 else ""
        print(f"  iter[{i:>2d}]  len={L:>4d}  {dt*1000:8.1f}ms{marker}")

    # ---------- SUMMARY ----------
    if not times:
        return
    first        = times[0]
    rest         = times[1:]
    print(f"\n{'='*55}")
    print(f"{'First-call latency':30}{first*1000:>15.1f}ms")
    if rest:
        print(f"{'Subsequent mean':30}{statistics.mean(rest)*1000:>15.1f}ms")
        print(f"{'Subsequent median':30}{statistics.median(rest)*1000:>15.1f}ms")
        print(f"{'Subsequent min':30}{min(rest)*1000:>15.1f}ms")
        print(f"{'Subsequent max':30}{max(rest)*1000:>15.1f}ms")
        print(f"{'First / median ratio':30}{first / statistics.median(rest):>16.2f}×")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
