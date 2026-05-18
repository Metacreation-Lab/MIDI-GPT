"""Export the pure-PT GPT-2 model to ONNX.

Two graphs are produced:
  - prefill: input_ids (B, T), no past_kv -> logits (B, T, V), present_kv
  - decode:  input_ids (B, 1), past_kv (n_layer × 2 × (B, H, T_past, Dh))
             -> logits (B, 1, V), present_kv

Dynamic axes: batch, sequence (prefill only), past_len (decode only).

Usage:
    .venv/bin/python3 midigpt_refactor/scripts/export_onnx.py \\
        --ckpt models/yellow_remapped.pt \\
        --out_prefill models/yellow_prefill.onnx \\
        --out_decode  models/yellow_decode.onnx
"""
import argparse
import torch
from midigpt_refactor.inference.model import GPT2LMHeadModel


class _PrefillWrapper(torch.nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, input_ids):
        logits, present = self.m(input_ids, past_kv=None)
        # Flatten present to a single tensor list for ONNX export simplicity
        flat = []
        for k, v in present:
            flat.append(k)
            flat.append(v)
        return (logits, *flat)


class _DecodeWrapper(torch.nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, input_ids, *flat_past):
        past_kv = tuple((flat_past[2*i], flat_past[2*i+1]) for i in range(len(flat_past)//2))
        logits, present = self.m(input_ids, past_kv=past_kv)
        flat = []
        for k, v in present:
            flat.append(k); flat.append(v)
        return (logits, *flat)


def export(ckpt: str, out_prefill: str, out_decode: str) -> None:
    model = GPT2LMHeadModel.from_torchscript(ckpt)
    model.eval()
    cfg = model.cfg
    n_layer, n_head, head_dim = cfg.n_layer, cfg.n_head, cfg.head_dim

    # ----- prefill -----
    pw = _PrefillWrapper(model).eval()
    dummy_ids = torch.zeros(1, 16, dtype=torch.long)
    present_names = [f"present_{i}_{kv}" for i in range(n_layer) for kv in ("k", "v")]
    torch.onnx.export(
        pw, (dummy_ids,), out_prefill,
        input_names=["input_ids"],
        output_names=["logits", *present_names],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "logits":    {0: "batch", 1: "seq"},
            **{n: {0: "batch", 2: "seq"} for n in present_names},
        },
        opset_version=17,
    )
    print(f"prefill → {out_prefill}")

    # ----- decode (T_in = 1, T_past = dynamic) -----
    dw = _DecodeWrapper(model).eval()
    dummy_id = torch.zeros(1, 1, dtype=torch.long)
    dummy_past = [torch.zeros(1, n_head, 8, head_dim) for _ in range(2 * n_layer)]
    past_names = [f"past_{i}_{kv}" for i in range(n_layer) for kv in ("k", "v")]
    torch.onnx.export(
        dw, (dummy_id, *dummy_past), out_decode,
        input_names=["input_ids", *past_names],
        output_names=["logits", *present_names],
        dynamic_axes={
            "input_ids": {0: "batch"},
            "logits":    {0: "batch"},
            **{n: {0: "batch", 2: "past_len"} for n in past_names},
            **{n: {0: "batch", 2: "total_len"} for n in present_names},
        },
        opset_version=17,
    )
    print(f"decode  → {out_decode}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_prefill", required=True)
    ap.add_argument("--out_decode", required=True)
    args = ap.parse_args()
    export(args.ckpt, args.out_prefill, args.out_decode)
