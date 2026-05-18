"""Pack a TorchScript checkpoint + encoder config into a self-contained `.pt`.

Output layout:
    {
        "format_version": 1,
        "config":         {vocab_size, n_positions, n_embd, n_layer, n_head},
        "encoder_config": {...},   # parsed from --encoder_config JSON
        "state_dict":     {...},   # legacy attn.bias/masked_bias stripped
    }

Loaders auto-detect this format via `GPT2LMHeadModel.from_pretrained(path)`.

Usage:
    .venv/bin/python3 midigpt_refactor/scripts/pack_checkpoint.py \\
        --ckpt models/yellow_remapped.pt \\
        --encoder_config models/yellow_config.json \\
        --out models/yellow.pt
"""
import argparse
import json
from midigpt_refactor.inference.model import GPT2LMHeadModel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Input TorchScript .pt")
    ap.add_argument("--encoder_config", required=True, help="Tokenizer config JSON")
    ap.add_argument("--out", required=True, help="Output packed .pt")
    args = ap.parse_args()

    with open(args.encoder_config) as f:
        enc_cfg = json.load(f)

    model = GPT2LMHeadModel.from_torchscript(args.ckpt, device="cpu")
    model.save_pretrained(args.out, encoder_config=enc_cfg)
    print(f"packed → {args.out}")
    print(f"  arch:    {model.cfg}")
    print(f"  encoder: {len(enc_cfg)} top-level keys")


if __name__ == "__main__":
    main()
