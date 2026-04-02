"""compare_models.py

Compare logits and output distributions between two TorchScript .pt models
on the same input token sequence to diagnose behavioural differences.

Usage:
    python compare_models.py \
        --model_a /path/to/model.pt \
        --model_b /path/to/other.pt \
        --midi /path/to/file.mid        # optional, uses synthetic tokens if absent
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "build"))
import midigpt  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_model(path):
    extra = {"metadata.json": ""}
    model = torch.jit.load(path, _extra_files=extra)
    meta = json.loads(extra["metadata.json"]) if extra["metadata.json"] else {}
    model.eval()
    return model, meta


def make_empty_pkv(batch_size, num_layers, num_heads, num_hidden):
    return tuple(
        (
            torch.zeros(batch_size, num_heads, 0, num_hidden),
            torch.zeros(batch_size, num_heads, 0, num_hidden),
        )
        for _ in range(num_layers)
    )


def get_tokens_from_midi(midi_path, encoder_name, num_bars=8):
    encoder_mode = midigpt.getEncoderType(encoder_name)
    tc = midigpt.TrainConfig()
    tc.num_bars = num_bars
    tc.min_tracks = 1
    tc.max_tracks = 12
    tc.use_microtiming = False
    tc.no_max_length = 0
    tc.resolution = 12

    encoder = midigpt.getEncoder(encoder_mode)
    tokens = encoder.midi_to_tokens(midi_path)
    return tokens  # list of ints


def get_synthetic_tokens(vocab_size, seq_len=256, seed=42):
    rng = np.random.default_rng(seed)
    return rng.integers(0, vocab_size, size=seq_len).tolist()


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def run_forward(model, tokens, meta):
    nl = meta.get("num_layers", 6)
    nh = meta.get("num_heads", 8)
    nhid = meta.get("num_hidden", 64)
    ids = torch.tensor([tokens], dtype=torch.long)
    pkv = make_empty_pkv(1, nl, nh, nhid)
    with torch.no_grad():
        logits, _ = model(ids, pkv)
    return logits.squeeze(0)  # [seq_len, vocab]


def analyse_logits(name, logits):
    probs = F.softmax(logits, dim=-1)
    entropy = -(probs * (probs + 1e-12).log()).sum(dim=-1)  # [seq_len]
    top1 = logits.argmax(dim=-1)  # [seq_len]
    top1_prob = probs.max(dim=-1).values

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  logits shape : {logits.shape}")
    print(f"  logits range : [{logits.min():.4f}, {logits.max():.4f}]")
    print(f"  logits mean  : {logits.mean():.4f}  std: {logits.std():.4f}")
    print(f"  entropy      : mean={entropy.mean():.4f}  min={entropy.min():.4f}  max={entropy.max():.4f}")
    print(f"  top-1 prob   : mean={top1_prob.mean():.4f}  min={top1_prob.min():.4f}  max={top1_prob.max():.4f}")
    print(f"  top-1 tokens : {top1[:20].tolist()}  ...")
    return logits, probs, entropy, top1


def compare(logits_a, logits_b, name_a, name_b):
    diff = (logits_a - logits_b).abs()
    print(f"\n{'='*60}")
    print(f"  DIFF: {name_a} vs {name_b}")
    print(f"{'='*60}")
    print(f"  |logits_a - logits_b|  max={diff.max():.6f}  mean={diff.mean():.6f}")

    # Are top-1 predictions the same?
    top1_a = logits_a.argmax(dim=-1)
    top1_b = logits_b.argmax(dim=-1)
    agree = (top1_a == top1_b).float().mean().item()
    print(f"  top-1 agreement      : {agree*100:.1f}%")

    # KL divergence A→B
    log_pa = F.log_softmax(logits_a, dim=-1)
    log_pb = F.log_softmax(logits_b, dim=-1)
    pa = log_pa.exp()
    pb = log_pb.exp()
    kl_ab = F.kl_div(log_pb, pa, reduction="batchmean").item()
    kl_ba = F.kl_div(log_pa, pb, reduction="batchmean").item()
    print(f"  KL(A||B)             : {kl_ab:.6f}")
    print(f"  KL(B||A)             : {kl_ba:.6f}")

    # Per-position max diff — show worst positions
    pos_max_diff = diff.max(dim=-1).values  # [seq_len]
    worst_pos = pos_max_diff.topk(5).indices.tolist()
    print(f"  worst positions (by max |diff|): {worst_pos}")
    for p in worst_pos[:3]:
        ta = logits_a[p].topk(3)
        tb = logits_b[p].topk(3)
        print(f"    pos {p:4d}: A top3={list(zip(ta.indices.tolist(), [f'{v:.3f}' for v in ta.values.tolist()]))}"
              f"  B top3={list(zip(tb.indices.tolist(), [f'{v:.3f}' for v in tb.values.tolist()]))}")


def check_forward_signature(model, name):
    print(f"\n{'='*60}")
    print(f"  FORWARD SIGNATURE: {name}")
    print(f"{'='*60}")
    try:
        schema = model.forward.schema
        print(f"  {schema}")
    except Exception as e:
        print(f"  (schema unavailable: {e})")
    try:
        code = model.forward.code
        # just print first 10 lines
        lines = code.strip().split('\n')
        for l in lines[:15]:
            print(f"  {l}")
        if len(lines) > 15:
            print(f"  ... ({len(lines)-15} more lines)")
    except Exception as e:
        print(f"  (code unavailable: {e})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_a", required=True)
    parser.add_argument("--model_b", required=True)
    parser.add_argument("--midi", default="")
    parser.add_argument("--num_bars", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=256,
                        help="Synthetic sequence length if no MIDI given")
    args = parser.parse_args()

    print(f"Loading {args.model_a} ...")
    model_a, meta_a = load_model(args.model_a)
    print(f"Loading {args.model_b} ...")
    model_b, meta_b = load_model(args.model_b)

    name_a = os.path.basename(args.model_a)
    name_b = os.path.basename(args.model_b)

    print(f"\nMeta A: {meta_a}")
    print(f"Meta B: {meta_b}")

    # ── Forward signatures ────────────────────────────────────────────────────
    check_forward_signature(model_a, name_a)
    check_forward_signature(model_b, name_b)

    # ── Token sequence ────────────────────────────────────────────────────────
    encoder_name = meta_a.get("encoder", meta_b.get("encoder", "EL_VELOCITY_DURATION_POLYPHONY_YELLOW_ENCODER"))
    encoder_mode = midigpt.getEncoderType(encoder_name)
    encoder = midigpt.getEncoder(encoder_mode)
    vocab_size = encoder.vocab_size()

    if args.midi:
        print(f"\nEncoding MIDI: {args.midi}")
        tokens = get_tokens_from_midi(args.midi, encoder_name, args.num_bars)
        print(f"  Got {len(tokens)} tokens from MIDI")
        if len(tokens) > 512:
            tokens = tokens[:512]
            print(f"  Truncated to 512")
    else:
        tokens = get_synthetic_tokens(vocab_size, args.seq_len)
        print(f"\nUsing {len(tokens)} synthetic random tokens (vocab_size={vocab_size})")

    # ── Forward pass ─────────────────────────────────────────────────────────
    print("\nRunning forward pass on model A ...")
    logits_a = run_forward(model_a, tokens, meta_a)
    print("Running forward pass on model B ...")
    logits_b = run_forward(model_b, tokens, meta_b)

    # ── Per-model analysis ────────────────────────────────────────────────────
    analyse_logits(name_a, logits_a)
    analyse_logits(name_b, logits_b)

    # ── Comparison ────────────────────────────────────────────────────────────
    # Align vocab dim (in case they differ)
    v = min(logits_a.shape[-1], logits_b.shape[-1])
    compare(logits_a[:, :v], logits_b[:, :v], name_a, name_b)

    # ── Checkpoint weight comparison (first layer) ────────────────────────────
    print(f"\n{'='*60}")
    print("  WEIGHT SPOT-CHECK (transformer.wte / model.transformer.wte)")
    print(f"{'='*60}")
    params_a = dict(model_a.named_parameters())
    params_b = dict(model_b.named_parameters())

    # Normalise names: strip leading 'model.' if present
    def norm(d):
        return {k.removeprefix("model."): v for k, v in d.items()}

    pa = norm(params_a)
    pb = norm(params_b)

    for key in ["transformer.wte.weight", "transformer.wpe.weight",
                "transformer.h.0.attn.c_attn.weight", "lm_head.weight"]:
        va = pa.get(key)
        vb = pb.get(key)
        if va is None and vb is None:
            print(f"  {key}: absent in both")
        elif va is None:
            print(f"  {key}: only in B, shape={vb.shape}")
        elif vb is None:
            print(f"  {key}: only in A, shape={va.shape}")
        else:
            diff = (va - vb).abs()
            print(f"  {key}: max_diff={diff.max():.8f}  mean_diff={diff.mean():.8f}")


if __name__ == "__main__":
    main()
