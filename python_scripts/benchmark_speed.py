"""benchmark_speed.py

Measure inference speed of TorchScript .pt models across:
  - Different sequence lengths
  - Prefill (full sequence, no cache) vs autoregressive decode (token-by-token, with cache)

Usage:
    python benchmark_speed.py \
        --model_a /path/to/model.pt \
        --model_b /path/to/other.pt \
        --seq_lens 32 64 128 256 512 1024 \
        --warmup 3 \
        --repeats 10 \
        --output benchmark_results.json
"""

import argparse
import json
import os
import sys
import time

import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "build"))
import midigpt  # noqa: E402


def load_model(path):
    extra = {"metadata.json": ""}
    model = torch.jit.load(path, _extra_files=extra)
    meta = json.loads(extra["metadata.json"]) if extra["metadata.json"] else {}
    model.eval()
    return model, meta


def make_empty_pkv(batch_size, num_layers, num_heads, num_hidden, device):
    return tuple(
        (
            torch.zeros(batch_size, num_heads, 0, num_hidden, device=device),
            torch.zeros(batch_size, num_heads, 0, num_hidden, device=device),
        )
        for _ in range(num_layers)
    )


def bench_prefill(model, seq_len, meta, device, warmup, repeats):
    """Prefill: feed full sequence, no cached tokens."""
    nl = meta.get("num_layers", 6)
    nh = meta.get("num_heads", 8)
    nhid = meta.get("num_hidden", 64)

    ids = torch.randint(0, 512, (1, seq_len), dtype=torch.long, device=device)
    pkv = make_empty_pkv(1, nl, nh, nhid, device)

    with torch.no_grad():
        for _ in range(warmup):
            model(ids, pkv)

    times = []
    with torch.no_grad():
        for _ in range(repeats):
            t0 = time.perf_counter()
            model(ids, pkv)
            times.append(time.perf_counter() - t0)

    return times


def bench_autoregressive(model, seq_len, meta, device, warmup, repeats):
    """Autoregressive decode: one token at a time, building KV cache."""
    nl = meta.get("num_layers", 6)
    nh = meta.get("num_heads", 8)
    nhid = meta.get("num_hidden", 64)

    def _run_once():
        pkv = make_empty_pkv(1, nl, nh, nhid, device)
        with torch.no_grad():
            for step in range(seq_len):
                tok = torch.randint(0, 512, (1, 1), dtype=torch.long, device=device)
                _, pkv = model(tok, pkv)

    for _ in range(warmup):
        _run_once()

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _run_once()
        times.append(time.perf_counter() - t0)

    return times


def summarise(times, label, seq_len):
    t = sorted(times)
    mean_ms = sum(t) / len(t) * 1000
    median_ms = t[len(t) // 2] * 1000
    min_ms = t[0] * 1000
    max_ms = t[-1] * 1000
    ms_per_tok = mean_ms / seq_len
    print(
        f"  {label:<55}  mean={mean_ms:7.1f}ms  median={median_ms:7.1f}ms"
        f"  min={min_ms:7.1f}ms  max={max_ms:7.1f}ms  ({ms_per_tok:.3f}ms/tok)"
    )
    return {"mean_ms": mean_ms, "median_ms": median_ms, "min_ms": min_ms,
            "max_ms": max_ms, "ms_per_tok": ms_per_tok}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_a", required=True)
    parser.add_argument("--model_b", required=True)
    parser.add_argument("--seq_lens", type=int, nargs="+",
                        default=[32, 64, 128, 256, 512])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", default="benchmark_results.json")
    parser.add_argument("--num_threads", type=int, default=0)
    args = parser.parse_args()

    num_threads = args.num_threads or int(os.getenv("SLURM_CPUS_PER_TASK", 1))
    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(1)

    device = torch.device(args.device)

    name_a = os.path.basename(args.model_a)
    name_b = os.path.basename(args.model_b)

    print(f"Loading model A: {name_a}")
    model_a, meta_a = load_model(args.model_a)
    model_a = model_a.to(device)

    print(f"Loading model B: {name_b}")
    model_b, meta_b = load_model(args.model_b)
    model_b = model_b.to(device)

    print(f"\nDevice  : {args.device}")
    print(f"Threads : {num_threads}")
    print(f"Warmup  : {args.warmup}  Repeats: {args.repeats}")
    print(f"Seq lens: {args.seq_lens}")
    print(f"\nMeta A  : {meta_a}")
    print(f"Meta B  : {meta_b}")
    print()

    results = {"model_a": name_a, "model_b": name_b,
               "device": args.device, "benchmarks": {}}

    for seq_len in args.seq_lens:
        print(f"\n{'='*80}")
        print(f"  seq_len = {seq_len}")
        print(f"{'='*80}")

        row = {}

        # ── Prefill (no cache) ────────────────────────────────────────────────
        print("  [PREFILL — full sequence, no cache]")
        t = bench_prefill(model_a, seq_len, meta_a, device, args.warmup, args.repeats)
        row["prefill_a"] = summarise(t, f"A  {name_a[:40]}", seq_len)

        t = bench_prefill(model_b, seq_len, meta_b, device, args.warmup, args.repeats)
        row["prefill_b"] = summarise(t, f"B  {name_b[:40]}", seq_len)

        ratio = row["prefill_b"]["mean_ms"] / row["prefill_a"]["mean_ms"]
        print(f"  → B / A ratio (prefill): {ratio:.3f}x  "
              f"({'faster' if ratio < 1 else 'slower'})")
        row["prefill_ratio_b_over_a"] = ratio

        # ── Autoregressive (with cache) ───────────────────────────────────────
        print("  [AUTOREGRESSIVE — token-by-token, with KV cache]")
        t = bench_autoregressive(model_a, seq_len, meta_a, device, args.warmup, args.repeats)
        row["ar_a"] = summarise(t, f"A  {name_a[:40]}", seq_len)

        t = bench_autoregressive(model_b, seq_len, meta_b, device, args.warmup, args.repeats)
        row["ar_b"] = summarise(t, f"B  {name_b[:40]}", seq_len)

        ratio_ar = row["ar_b"]["mean_ms"] / row["ar_a"]["mean_ms"]
        print(f"  → B / A ratio (autoregressive): {ratio_ar:.3f}x  "
              f"({'faster' if ratio_ar < 1 else 'slower'})")
        row["ar_ratio_b_over_a"] = ratio_ar

        results["benchmarks"][str(seq_len)] = row

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n\n{'='*80}")
    print("  SUMMARY TABLE")
    print(f"{'='*80}")
    header = f"  {'seq_len':>8}  {'prefill_A(ms)':>14}  {'prefill_B(ms)':>14}  {'ratio':>7}  {'ar_A(ms)':>10}  {'ar_B(ms)':>10}  {'ratio':>7}"
    print(header)
    print("  " + "-"*78)
    for seq_len in args.seq_lens:
        r = results["benchmarks"][str(seq_len)]
        print(
            f"  {seq_len:>8}"
            f"  {r['prefill_a']['mean_ms']:>14.1f}"
            f"  {r['prefill_b']['mean_ms']:>14.1f}"
            f"  {r['prefill_ratio_b_over_a']:>7.3f}x"
            f"  {r['ar_a']['mean_ms']:>10.1f}"
            f"  {r['ar_b']['mean_ms']:>10.1f}"
            f"  {r['ar_ratio_b_over_a']:>7.3f}x"
        )
    print()
    print(f"  A = {name_a}")
    print(f"  B = {name_b}")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
