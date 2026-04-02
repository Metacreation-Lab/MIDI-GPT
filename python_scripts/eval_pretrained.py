"""eval_pretrained.py

Evaluate a single TorchScript .pt model on the validation (split 1) and test
(split 2) splits of a midigpt .arr dataset. Reports cross-entropy loss
comparable to HuggingFace Trainer eval logs from ongoing training runs, and
logs to Weights & Biases when WANDB_API_KEY is set.

Usage:
    python eval_pretrained.py \
        --model /path/to/model.pt \
        --num_bars 4 \
        --dataset /path/to/data.arr \
        --splits 1 2 \
        --num_batches 300 \
        --output eval_results.json

Notes:
  - Encoder type is read from the .pt metadata (embedded at conversion time).
  - --num_bars must match the value used during training:
      * model.pt (EL_VELOCITY_DURATION_POLYPHONY_YELLOW_ENCODER): set explicitly
      * steinberg_v2 (STEINBERG_W_P_C_S_ENCODER): 4 (from model filename)
  - Loss is token-level cross-entropy (matching HuggingFace Trainer).
  - Models are CPU-traced; evaluation runs on CPU only.
  - PyTorch intra-op thread count is set from SLURM_CPUS_PER_TASK (or
    --num_threads) so all allocated CPUs are used for OpenMP parallelism.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "build"))
import midigpt  # noqa: E402

from dotenv import load_dotenv
load_dotenv(os.path.join(_REPO, ".env"))

SPLIT_NAMES = {0: "train", 1: "val", 2: "test"}

DEFAULT_DATASET = os.path.join(
    os.path.expandvars("$HOME"),
    "scratch/.midigpt/datasets/EXPRESSIVE_ENCODER_NUM_BARS=4_RESOLUTION_12.arr",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_model_and_metadata(path: str):
    extra = {"metadata.json": ""}
    model = torch.jit.load(path, _extra_files=extra)
    meta = json.loads(extra["metadata.json"])
    return model, meta


def eval_split(model, jagged, encoder_mode, tc, split_id: int,
               batch_size: int, num_batches: int, pad_value: int = -100,
               num_layers: int = 6, num_heads: int = 8, num_hidden: int = 64):
    """Evaluate one split; return (mean_loss, total_tokens)."""
    total_loss = 0.0
    total_tokens = 0

    for batch_idx in range(num_batches):
        input_ids_raw, mask_raw = jagged.read_batch(
            batch_size, split_id, encoder_mode, tc
        )
        input_ids = torch.tensor(np.array(input_ids_raw), dtype=torch.long)
        mask = torch.tensor(np.array(mask_raw), dtype=torch.long)

        labels = input_ids.clone()
        labels[mask == 0] = pad_value

        # TorchScript models are traced with a concrete past_key_values tuple
        # (one (key, value) pair per layer). Passing () causes a type error
        # because TorchScript expects exactly num_layers pairs. We pass
        # zero-sequence-length tensors so no cached tokens are added.
        bsz = input_ids.shape[0]
        empty_pkv = tuple(
            (
                torch.zeros(bsz, num_heads, 0, num_hidden),
                torch.zeros(bsz, num_heads, 0, num_hidden),
            )
            for _ in range(num_layers)
        )

        with torch.no_grad():
            logits, _ = model(input_ids, empty_pkv)

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        valid = (shift_labels != pad_value).sum().item()
        if valid == 0:
            continue

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=pad_value,
            reduction="sum",
        )
        total_loss += loss.item()
        total_tokens += valid

        if (batch_idx + 1) % 50 == 0:
            cur = total_loss / total_tokens
            split_name = SPLIT_NAMES.get(split_id, split_id)
            print(f"    [{split_name}] batch {batch_idx + 1}/{num_batches}  loss={cur:.4f}",
                  flush=True)

    mean_loss = total_loss / total_tokens if total_tokens > 0 else float("nan")
    return mean_loss, total_tokens


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a TorchScript .pt model on val/test splits"
    )
    parser.add_argument("--model", required=True,
                        help="Path to TorchScript .pt model file")
    parser.add_argument("--dataset", default=DEFAULT_DATASET,
                        help=f"Path to .arr dataset (default: {DEFAULT_DATASET})")
    parser.add_argument("--num_bars", type=int, default=4,
                        help="num_bars — must match the training config (default: 4)")
    parser.add_argument("--resolution", type=int, default=12,
                        help="Encoding resolution for TrainConfig (default: 12)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size (default: 32)")
    parser.add_argument("--splits", type=int, nargs="+", default=[1, 2],
                        help="Split IDs: 1=val, 2=test (default: 1 2)")
    parser.add_argument("--num_batches", type=int, default=300,
                        help="Random batches per split (default: 300 ≈ 9,600 samples)")
    parser.add_argument("--num_threads", type=int, default=0,
                        help="PyTorch intra-op threads (0 = auto from SLURM_CPUS_PER_TASK)")
    parser.add_argument("--output", default="eval_pretrained_results.json",
                        help="Output JSON file (default: eval_pretrained_results.json)")
    parser.add_argument("--wandb_project", type=str, default="midi-gpt",
                        help="W&B project name (default: midi-gpt)")
    parser.add_argument("--wandb_group", type=str, default="eval_pretrained",
                        help="W&B run group (default: eval_pretrained)")
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable W&B logging even if WANDB_API_KEY is set")
    args = parser.parse_args()

    # ── Thread count: use SLURM allocation if not overridden ─────────────────
    num_threads = args.num_threads or int(os.getenv("SLURM_CPUS_PER_TASK", 1))
    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(1)

    # ── Load model ────────────────────────────────────────────────────────────
    model_name = os.path.basename(args.model)
    print(f"Model   : {model_name}")
    model, meta = load_model_and_metadata(args.model)
    encoder_name = meta.get("encoder", "UNKNOWN")
    arch_str = f"{meta.get('num_layers')}L_{meta.get('num_heads')}H"
    print(f"Encoder : {encoder_name}")
    print(f"Arch    : {arch_str}")
    print(f"Dataset : {args.dataset}")
    print(f"Splits  : {[SPLIT_NAMES.get(s, s) for s in args.splits]}")
    print(f"Batches : {args.num_batches} per split")
    print(f"Threads : {num_threads}")
    print()
    model.eval()

    # ── W&B init ──────────────────────────────────────────────────────────────
    use_wandb = not args.no_wandb and bool(os.getenv("WANDB_API_KEY"))
    wb_run = None
    if use_wandb:
        import wandb
        wb_run = wandb.init(
            project=args.wandb_project,
            name=model_name,
            group=args.wandb_group,
            config={
                "encoder": encoder_name,
                "num_bars": args.num_bars,
                "resolution": args.resolution,
                "batch_size": args.batch_size,
                "num_batches": args.num_batches,
                "num_threads": num_threads,
                "arch": arch_str,
                "model_path": args.model,
            },
        )
        print(f"W&B run : {wb_run.url}\n")

    # ── Dataset / encoder setup ───────────────────────────────────────────────
    encoder_mode = midigpt.getEncoderType(encoder_name)

    tc = midigpt.TrainConfig()
    tc.num_bars = args.num_bars
    tc.min_tracks = 1
    tc.max_tracks = 12
    tc.use_microtiming = False
    tc.no_max_length = 0
    tc.resolution = args.resolution

    jagged = midigpt.Jagged(args.dataset)
    jagged.set_num_bars(args.num_bars)
    jagged.set_min_tracks(1)
    jagged.set_max_tracks(12)
    jagged.set_max_seq_len(2048)
    jagged.set_seed(42)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    results = {}
    wandb_log = {}

    for split_id in args.splits:
        split_name = SPLIT_NAMES.get(split_id, str(split_id))
        split_size = jagged.get_split_size(split_id)
        print(f"Split {split_id} ({split_name}, size={split_size}) ...")

        t0 = time.time()
        loss, n_tokens = eval_split(
            model, jagged, encoder_mode, tc,
            split_id, args.batch_size, args.num_batches,
            num_layers=meta.get("num_layers", 6),
            num_heads=meta.get("num_heads", 8),
            num_hidden=meta.get("num_hidden", 64),
        )
        elapsed = time.time() - t0

        print(f"  → {split_name} loss = {loss:.4f}  "
              f"({n_tokens:,} tokens, {args.num_batches} batches, {elapsed:.1f}s)\n")

        results[split_name] = loss
        wandb_log[f"eval/{split_name}_loss"] = loss

    if use_wandb and wb_run is not None:
        wb_run.log(wandb_log)
        wb_run.finish()

    # ── Output ────────────────────────────────────────────────────────────────
    output = {
        model_name: {
            "encoder": encoder_name,
            "num_bars": args.num_bars,
            "arch": arch_str,
            "losses": results,
        }
    }

    print("=" * 50)
    for split_name, loss in results.items():
        print(f"  {split_name:>6} loss: {loss:.4f}")

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
