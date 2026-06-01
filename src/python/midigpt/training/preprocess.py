"""Pre-compute and cache valid indices for a GigaMIDI parquet shard.

Run this once per (parquet, encoder) combination before training so the
dataset init is instant (cache hit) rather than running the subprocess scan
at training startup.

Usage:
    python -m midigpt.training.preprocess \\
        --parquet /data/v2.0.0/train/00000.parquet \\
        --checkpoint models/yellow.pt \\
        [--min-bars 4] [--min-tracks 1]

    # Or with a raw encoder-config JSON instead of a checkpoint bundle:
    python -m midigpt.training.preprocess \\
        --parquet /data/v2.0.0/train/00000.parquet \\
        --encoder-config models/yellow_encoder.json

Multiple parquet shards can be given at once:
    python -m midigpt.training.preprocess \\
        --parquet /data/v2.0.0/train/*.parquet \\
        --checkpoint models/yellow.pt
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path


def _load_encoder_config(checkpoint: str | None, encoder_config: str | None) -> dict:
    if checkpoint:
        import torch

        data = torch.load(checkpoint, map_location="cpu", weights_only=False)
        enc = data.get("encoder_config", {})
        return enc if isinstance(enc, dict) else json.loads(enc)
    if encoder_config:
        with open(encoder_config) as f:
            return json.load(f)
    return {}


def preprocess(
    parquet_paths: list[str],
    checkpoint: str | None = None,
    encoder_config: str | None = None,
    min_bars: int = 4,
    min_tracks: int = 1,
) -> None:
    from midigpt.training.dataset import _load_or_build_valid_indices

    cfg = _load_encoder_config(checkpoint, encoder_config)
    ts_list = cfg.get("time_signatures")
    valid_ts: frozenset[str] | None = frozenset(ts_list) if ts_list else None

    num_bars_map = cfg.get("num_bars_map") or [min_bars]
    effective_min_bars = min(min(int(x) for x in num_bars_map), min_bars)

    print(f"Encoder time signatures: {len(valid_ts) if valid_ts else 'all (no check)'}")
    print(f"Filter params: min_bars={effective_min_bars}, min_tracks={min_tracks}")
    print()

    total_kept = 0
    for path in parquet_paths:
        print(f"[{path}]")
        t0 = time.time()
        valid = _load_or_build_valid_indices(
            path,
            min_bars=effective_min_bars,
            min_tracks=min_tracks,
            valid_time_sigs=valid_ts,
        )
        elapsed = time.time() - t0
        total_kept += len(valid)
        print(f"  → {len(valid)} valid rows in {elapsed:.1f}s\n")

    print(f"Total valid rows across {len(parquet_paths)} shard(s): {total_kept}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-compute valid-index cache for GigaMIDI parquet shards."
    )
    parser.add_argument(
        "--parquet",
        nargs="+",
        required=True,
        metavar="PATH",
        help="Parquet shard(s). Supports shell globs if quoted.",
    )
    parser.add_argument(
        "--checkpoint",
        metavar="PATH",
        help="Packed .pt bundle (encoder config + weights). Mutually exclusive with --encoder-config.",
    )
    parser.add_argument(
        "--encoder-config",
        metavar="PATH",
        help="Raw encoder config JSON. Alternative to --checkpoint.",
    )
    parser.add_argument("--min-bars", type=int, default=4)
    parser.add_argument("--min-tracks", type=int, default=1)
    args = parser.parse_args()

    if args.checkpoint and args.encoder_config:
        parser.error("Use --checkpoint or --encoder-config, not both.")

    # Expand globs (useful when the shell doesn't expand them, e.g. via SLURM)
    paths: list[str] = []
    for pattern in args.parquet:
        expanded = sorted(glob.glob(pattern))
        paths.extend(expanded if expanded else [pattern])

    missing = [p for p in paths if not Path(p).exists()]
    if missing:
        print(f"Error: file(s) not found: {missing}", file=sys.stderr)
        sys.exit(1)

    preprocess(
        parquet_paths=paths,
        checkpoint=args.checkpoint,
        encoder_config=args.encoder_config,
        min_bars=args.min_bars,
        min_tracks=args.min_tracks,
    )


if __name__ == "__main__":
    main()
