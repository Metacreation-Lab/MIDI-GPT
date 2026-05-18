"""Remap a Yellow-era TorchScript checkpoint from orig vocab IDs to ref vocab IDs.

The original midigpt and midigpt_refactor cover the same 647 tokens, but assign
them different integer IDs. Loading `models/yellow.pt` with the refactor vocab
produces garbage notes because `wte`/`lm_head` rows are indexed by orig IDs.

This script permutes the rows of those two tensors so the checkpoint speaks
ref-vocab IDs, then re-saves as a TorchScript module.

Usage:
    .venv/bin/python3 midigpt_refactor/scripts/remap_checkpoint.py \\
        --ckpt models/yellow.pt \\
        --config models/yellow_config.json \\
        --out models/yellow_remapped.pt
"""
import argparse
import torch
import midigpt
from midigpt_refactor import _core
from midigpt_refactor.tokenizer.tokenizer import Tokenizer
from midigpt_refactor.compat import (
    build_orig_to_ref_mapping, remap_embedding_weight,
)


def remap_checkpoint(ckpt_path: str, config_path: str, out_path: str) -> None:
    cfg = _core.EncoderConfig.from_json(open(config_path).read())
    ref_vocab = Tokenizer(cfg)._vocab
    orig_enc = midigpt.ElVelocityDurationPolyphonyYellowEncoder()

    mapping = build_orig_to_ref_mapping(orig_enc, ref_vocab)
    n_mapped = sum(1 for m in mapping if m >= 0)
    print(f"mapping: {n_mapped}/{len(mapping)} orig IDs → ref IDs")

    model = torch.jit.load(ckpt_path, map_location="cpu")
    model.eval()

    wte = model.transformer.wte.weight
    head = model.lm_head.weight
    print(f"wte:      {tuple(wte.shape)}")
    print(f"lm_head:  {tuple(head.shape)}")

    new_wte = remap_embedding_weight(wte.detach(), mapping, ref_vocab.size())
    new_head = remap_embedding_weight(head.detach(), mapping, ref_vocab.size())

    with torch.no_grad():
        model.transformer.wte.weight.copy_(new_wte)
        model.lm_head.weight.copy_(new_head)

    torch.jit.save(model, out_path)
    print(f"saved → {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    remap_checkpoint(args.ckpt, args.config, args.out)
