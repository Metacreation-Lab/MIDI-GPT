"""Reconstruct an orig-vocab TorchScript checkpoint from a ref-remapped one.

Inverts the row permutation done by `remap_checkpoint.py` and injects
`extra/metadata.json` so the file is loadable by the original midigpt C++ JIT
loader.

orig and ref vocab sizes are identical (647); only the row order differs, so
the underlying ScriptModule's parameter shapes are preserved and we can do
an in-place tensor swap.

Usage:
    .venv/bin/python3 midigpt_refactor/scripts/reverse_remap_checkpoint.py \\
        --ckpt models/yellow_remapped.pt \\
        --config models/yellow_config.json \\
        --encoder EL_VELOCITY_DURATION_POLYPHONY_YELLOW_ENCODER \\
        --num_heads 8 --num_hidden 64 --num_layers 6 \\
        --out models/yellow_orig_restored.pt
"""
import argparse
import json
import torch
import midigpt
from midigpt_refactor import _core
from midigpt_refactor.tokenizer.tokenizer import Tokenizer
from midigpt_refactor.compat import build_orig_to_ref_mapping


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="ref-remapped TorchScript .pt")
    ap.add_argument("--config", required=True, help="yellow_config.json")
    ap.add_argument("--encoder", required=True, help="C++ encoder enum string")
    ap.add_argument("--num_heads", type=int, required=True)
    ap.add_argument("--num_hidden", type=int, required=True)
    ap.add_argument("--num_layers", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = _core.EncoderConfig.from_json(open(args.config).read())
    ref_vocab = Tokenizer(cfg)._vocab
    orig_enc = midigpt.ElVelocityDurationPolyphonyYellowEncoder()

    mapping = build_orig_to_ref_mapping(orig_enc, ref_vocab)
    orig_V = orig_enc.vocab_size()
    if orig_V != ref_vocab.size():
        raise RuntimeError(f"vocab size mismatch orig={orig_V} ref={ref_vocab.size()}")

    model = torch.jit.load(args.ckpt, map_location="cpu")
    model.eval()

    wte = model.transformer.wte.weight.detach().clone()
    head = model.lm_head.weight.detach().clone()
    new_wte = torch.zeros_like(wte)
    new_head = torch.zeros_like(head)
    for orig_id, ref_id in enumerate(mapping):
        if ref_id >= 0:
            new_wte[orig_id] = wte[ref_id]
            new_head[orig_id] = head[ref_id]

    with torch.no_grad():
        model.transformer.wte.weight.copy_(new_wte)
        model.lm_head.weight.copy_(new_head)

    metadata = {
        "encoder": args.encoder,
        "num_heads": args.num_heads,
        "num_hidden": args.num_hidden,
        "num_layers": args.num_layers,
        "model_dim": -1,
        "new_state": True,
        "traced_device": "cpu",
    }
    torch.jit.save(model, args.out, _extra_files={"metadata.json": json.dumps(metadata)})
    print(f"saved → {args.out}")
    print(f"  metadata: {metadata}")


if __name__ == "__main__":
    main()
