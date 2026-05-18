"""Benchmark all inference backends on the same prefill/decode workload.

Reports per-backend median wall-clock for:
    - prefill at ctx ∈ {128, 512, 1024, 2048}
    - decode-step (T=1) at past_len ∈ {128, 512, 1024}

Parity: each backend's logits are compared to the TorchScript baseline
(max abs diff + top-1 agreement on the final position).

Usage:
    .venv/bin/python3 midigpt_refactor/scripts/bench_runtimes.py \\
        --ckpt models/yellow_remapped.pt
"""
import argparse
import statistics
import time
import traceback
import torch

from midigpt_refactor.inference.runtimes import (
    ALL_BACKENDS, TorchScriptBackend, _empty_past,
)


PREFILL_CTXS = (128, 512, 1024, 2048)
DECODE_PASTS = (128, 512, 1024)
BATCHES = (1, 2, 4, 8)
REPEATS_PREFILL = 3
REPEATS_DECODE = 10
WARMUP_PREFILL = 1
WARMUP_DECODE = 2


def _bench(fn, repeats: int, warmup: int) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts), min(ts)


def _make_past(cfg, past_len: int, batch: int = 1):
    return tuple(
        (torch.zeros(batch, cfg.n_head, past_len, cfg.head_dim),
         torch.zeros(batch, cfg.n_head, past_len, cfg.head_dim))
        for _ in range(cfg.n_layer)
    )


def _run_backend(cls, ckpt: str, baseline_prefill: dict | None):
    name = cls.name
    avail, reason = cls.is_available()
    if not avail:
        return {"name": name, "status": "SKIP", "reason": reason, "rows": []}

    try:
        b = cls()
        t0 = time.perf_counter()
        b.setup(ckpt)
        setup_ms = (time.perf_counter() - t0) * 1000
    except NotImplementedError as e:
        return {"name": name, "status": "STUB", "reason": str(e), "rows": []}
    except Exception as e:
        return {"name": name, "status": "ERROR", "reason": f"setup: {type(e).__name__}: {e}", "rows": []}

    cfg = b.cfg
    rows = []
    parity = []

    # ---- prefill bench ----
    for B in BATCHES:
        for ctx in PREFILL_CTXS:
            if ctx > cfg.n_positions:
                continue
            torch.manual_seed(ctx)
            ids = torch.randint(0, cfg.vocab_size, (B, ctx))
            def f():
                b.forward(ids, past_kv=None)
            try:
                med, mn = _bench(f, REPEATS_PREFILL, WARMUP_PREFILL)
                rows.append(("prefill", ctx, B, med * 1000, mn * 1000))
            except Exception as e:
                rows.append(("prefill", ctx, B, None, None))
                parity.append(f"ctx={ctx} B={B}: BENCH ERROR {type(e).__name__}: {e}")
                continue
            # parity only at B=1 (baseline is B=1)
            if B == 1:
                try:
                    with torch.no_grad():
                        logits, _ = b.forward(ids, past_kv=None)
                    if baseline_prefill and ctx in baseline_prefill:
                        ref_logits = baseline_prefill[ctx]
                        diff = (logits.float() - ref_logits.float()).abs().max().item()
                        agree = (logits.argmax(-1) == ref_logits.argmax(-1)).float().mean().item()
                        parity.append(f"ctx={ctx}: diff={diff:.2e} agree={agree*100:.1f}%")
                except Exception as e:
                    parity.append(f"ctx={ctx}: PARITY ERROR {type(e).__name__}: {e}")

    # ---- decode bench ----
    for B in BATCHES:
        for past_len in DECODE_PASTS:
            if past_len + 1 > cfg.n_positions:
                continue
            ids = torch.randint(0, cfg.vocab_size, (B, 1))
            past = _make_past(cfg, past_len, B)
            def f():
                b.forward(ids, past_kv=past)
            try:
                med, mn = _bench(f, REPEATS_DECODE, WARMUP_DECODE)
                rows.append(("decode", past_len, B, med * 1000, mn * 1000))
            except Exception as e:
                rows.append(("decode", past_len, B, None, None))
                parity.append(f"decode past={past_len} B={B}: ERROR {type(e).__name__}: {e}")

    return {
        "name": name, "status": "OK", "setup_ms": setup_ms,
        "rows": rows, "parity": parity, "reason": "",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    args = ap.parse_args()

    # Compute baseline logits (TorchScript) once for parity checking
    print("[baseline] running TorchScript for parity reference...")
    baseline = TorchScriptBackend()
    baseline.setup(args.ckpt)
    baseline_prefill: dict[int, torch.Tensor] = {}
    for ctx in PREFILL_CTXS:
        if ctx > baseline.cfg.n_positions: continue
        torch.manual_seed(ctx)
        ids = torch.randint(0, baseline.cfg.vocab_size, (1, ctx))
        with torch.no_grad():
            logits, _ = baseline.forward(ids)
        baseline_prefill[ctx] = logits

    # Reset seed so each backend sees same inputs
    results = []
    for cls in ALL_BACKENDS:
        torch.manual_seed(42)
        print(f"\n[{cls.name}] ...", flush=True)
        r = _run_backend(cls, args.ckpt, baseline_prefill)
        results.append(r)
        if r["status"] == "OK":
            print(f"  setup: {r['setup_ms']:.1f}ms")
            for kind, n, B, med, mn in r["rows"]:
                ms = f"{med:.2f}ms (min {mn:.2f})" if med is not None else "FAIL"
                print(f"  {kind:8s} n={n:<5d} B={B}  {ms}")
            for p in r["parity"]:
                print(f"  parity {p}")
        else:
            print(f"  [{r['status']}] {r['reason']}")

    # ---- summary tables (one per batch) ----
    for B in BATCHES:
        print("\n" + "=" * 110)
        print(f"PREFILL latency (ms median, B={B} × N → logits)")
        print(f"{'backend':<24} " + " ".join(f"{c:>10d}" for c in PREFILL_CTXS))
        print("-" * 110)
        for r in results:
            if r["status"] != "OK": continue
            cells = []
            for ctx in PREFILL_CTXS:
                v = next((m for k, n, b_, m, _ in r["rows"]
                         if k == "prefill" and n == ctx and b_ == B), None)
                cells.append(f"{v:>10.2f}" if v is not None else f"{'n/a':>10}")
            print(f"{r['name']:<24} " + " ".join(cells))

    for B in BATCHES:
        print("\n" + "=" * 110)
        print(f"DECODE step latency (ms median, B={B} × 1 with past_len)")
        print(f"{'backend':<24} " + " ".join(f"{c:>10d}" for c in DECODE_PASTS))
        print("-" * 110)
        for r in results:
            if r["status"] != "OK": continue
            cells = []
            for p in DECODE_PASTS:
                v = next((m for k, n, b_, m, _ in r["rows"]
                         if k == "decode" and n == p and b_ == B), None)
                cells.append(f"{v:>10.2f}" if v is not None else f"{'n/a':>10}")
            print(f"{r['name']:<24} " + " ".join(cells))

    print("\n" + "=" * 110)
    print("SKIPPED / STUB / ERROR")
    for r in results:
        if r["status"] != "OK":
            print(f"  [{r['status']:<5}] {r['name']:<24} {r['reason']}")
    print("=" * 110)


if __name__ == "__main__":
    main()
