"""Convert a trained HuggingFace GPT-2 checkpoint to TorchScript (.pt).

Embeds model metadata (encoder type, architecture) into the .pt file so the
C++ inference engine can load it without separate config files.

Transformers v5 uses DynamicCache internally, which is a Python object.
TorchScript tracing cannot serialize Python objects, and passing a raw tuple
hits AttributeError on .get_seq_length() inside GPT2Model.forward().
Fix: GPT2ManualWrapper reimplements the GPT-2 forward using raw weights and
pure tensor ops, bypassing the Cache API entirely. Verified bit-exact match
against the HuggingFace model output.

Usage:
    # From a trained checkpoint:
    python convert.py --ckpt_path /path/to/checkpoint-N \
                      --output model.pt \
                      --encoder EXPRESSIVE_ENCODER

    # From scratch (random weights):
    python convert.py --init --config config.json \
                      --output model.pt \
                      --encoder EXPRESSIVE_ENCODER

    # Inject metadata into existing .pt:
    python convert.py --inject --ckpt_path model.pt \
                      --metadata_path metadata.json \
                      --encoder EXPRESSIVE_ENCODER --new_state
"""

import json
import os
import sys

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Config

try:
    from transformers.modeling_utils import Conv1D
except ImportError:
    from transformers.pytorch_utils import Conv1D

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from custom_models import GPT2LMHeadModelCont, GPT2LMHeadModelContConfig
except ImportError:
    GPT2LMHeadModelCont = None
    GPT2LMHeadModelContConfig = None

import midigpt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _conv1d_to_linear(module):
    in_size, out_size = module.weight.shape
    linear = nn.Linear(in_size, out_size)
    linear.weight.data = module.weight.data.T.contiguous()
    linear.bias.data = module.bias.data
    return linear


def conv1d_to_linear(model):
    for name in list(model._modules):
        module = model._modules[name]
        if isinstance(module, Conv1D):
            model._modules[name] = _conv1d_to_linear(module)
        else:
            conv1d_to_linear(module)


def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print("Size (MB):", os.path.getsize("temp.p") / 1e6)
    os.remove("temp.p")


def quantize_model(model):
    conv1d_to_linear(model)
    return torch.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )


def prune_model(model):
    import torch.nn.utils.prune as prune

    conv1d_to_linear(model)
    for _, module in model.named_modules():
        if isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name="weight", amount=0.8)
            prune.remove(module, "weight")
    return model


# ---------------------------------------------------------------------------
# TorchScript wrapper — manual GPT-2 forward (no Cache API)
# ---------------------------------------------------------------------------
#
# Transformers v5 uses DynamicCache internally and calls
# past_key_values.get_seq_length() in GPT2Model.forward().  TorchScript
# tracing cannot handle Python objects (DynamicCache), and passing a raw
# tuple hits AttributeError on .get_seq_length().
#
# Solution: re-implement the GPT-2 forward with pure tensor ops so the
# transformers Cache API is never called.  All sizes come from .size()
# on tensors, keeping shapes dynamic in the traced graph.

import torch.nn.functional as F


def _attn_forward(block, hidden, past_k, past_v, n_head, n_embd):
    """One GPT-2 block: multi-head self-attention using a tuple KV cache.

    Transformers 5.x uses SDPA internally with no bias buffer.  We compute the
    causal mask explicitly from tensor shapes so it stays dynamic in the trace.
    """
    B, T, C = hidden.shape
    head_dim = n_embd // n_head
    past_len = past_k.size(2)

    # QKV via Conv1D (Conv1D weight layout: [n_embd, 3*n_embd])
    # F.linear expects weight [out, in], so transpose to match F.linear convention.
    qkv = F.linear(hidden, block.attn.c_attn.weight.T, block.attn.c_attn.bias)
    q, k, v = qkv.split(n_embd, dim=2)

    q = q.view(B, T, n_head, head_dim).transpose(1, 2)  # [B, nh, T, hd]
    k = k.view(B, T, n_head, head_dim).transpose(1, 2)
    v = v.view(B, T, n_head, head_dim).transpose(1, 2)

    # Concat KV cache
    k = torch.cat([past_k, k], dim=2)
    v = torch.cat([past_v, v], dim=2)

    # Causal additive mask [T, S]: 0 = attend, -inf = masked.
    # Queries can attend to ALL past keys and causally to current keys.
    # Current-to-current block: upper triangular is masked.
    # Uses tensor ops so shapes stay dynamic in the TorchScript graph.
    current_mask = torch.triu(
        torch.full((T, T), float("-inf"), device=hidden.device, dtype=hidden.dtype),
        diagonal=1,
    )
    past_mask = torch.zeros(T, past_len, device=hidden.device, dtype=hidden.dtype)
    attn_mask = torch.cat([past_mask, current_mask], dim=1)  # [T, S]

    # SDPA handles the 1/sqrt(head_dim) scaling internally.
    a = F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=attn_mask.unsqueeze(0).unsqueeze(0),
        dropout_p=0.0,
    )

    a = a.transpose(1, 2).contiguous().view(B, T, C)
    a = F.linear(a, block.attn.c_proj.weight.T, block.attn.c_proj.bias)
    return a, k, v


class GPT2ManualWrapper(nn.Module):
    """GPT-2 forward re-implemented with pure tensor ops.

    Accepts past_key_values as a plain tuple of (k, v) pairs (one per layer)
    and returns (logits, new_past_key_values) in the same format.
    The transformers Cache API is never called, so TorchScript tracing works.
    """

    def __init__(self, model: GPT2LMHeadModel):
        super().__init__()
        t = model.transformer
        self.wte = t.wte
        self.wpe = t.wpe
        self.drop = t.drop
        self.h = t.h
        self.ln_f = t.ln_f
        self.lm_head = model.lm_head
        self.n_head = model.config.n_head
        self.n_embd = model.config.n_embd

    def forward(self, input_ids, past_key_values):
        past_len = past_key_values[0][0].size(2)
        B, T = input_ids.shape[0], input_ids.shape[1]

        pos = torch.arange(past_len, past_len + T, device=input_ids.device, dtype=torch.long)
        hidden = self.wte(input_ids) + self.wpe(pos)
        hidden = self.drop(hidden)

        new_pkv = []
        for i in range(len(self.h)):
            block = self.h[i]
            past_k, past_v = past_key_values[i]

            # Attention sub-layer
            ln1 = block.ln_1(hidden)
            a_out, new_k, new_v = _attn_forward(block, ln1, past_k, past_v, self.n_head, self.n_embd)
            hidden = hidden + a_out

            # MLP sub-layer (Conv1D weights: [in, out])
            ln2 = block.ln_2(hidden)
            fc = F.linear(ln2, block.mlp.c_fc.weight.T, block.mlp.c_fc.bias)
            fc = block.mlp.act(fc)
            fc = F.linear(fc, block.mlp.c_proj.weight.T, block.mlp.c_proj.bias)
            hidden = hidden + fc

            new_pkv.append((new_k, new_v))

        hidden = self.ln_f(hidden)
        logits = self.lm_head(hidden)
        return logits, tuple(new_pkv)


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def convert(model, path, quantize=False, prune=False, force=False,
            control=False, ckpt_path=None, encoderX=None, device="cpu"):
    if os.path.exists(path) and not force:
        print(f"Output {path} already exists. Use --force to overwrite.")
        return

    # Trace on the specified device.  TorchScript bakes device literals into
    # the graph, so the C++ inference engine must load the model on the same
    # device that was used for tracing.
    device = torch.device(device)
    model = model.to(device)
    model.eval()
    if quantize:
        model = quantize_model(model)
    if prune:
        model = prune_model(model)
    print_size_of_model(model)

    cfg = model.config
    num_layers = cfg.n_layer
    num_heads = cfg.n_head
    num_hidden = cfg.n_embd // cfg.n_head

    # Empty past_key_values: shape [1, n_head, 0, head_dim]
    empty_pkv = tuple(
        (
            torch.zeros(1, num_heads, 0, num_hidden, device=device),
            torch.zeros(1, num_heads, 0, num_hidden, device=device),
        )
        for _ in range(num_layers)
    )

    # Trace with a 4-token prefix + empty cache (prefill scenario).
    # GPT2ManualWrapper uses only tensor ops and .size() calls, so shapes
    # remain dynamic in the traced graph — both prefill and single-token
    # generation steps work correctly at runtime.
    example_input = torch.zeros(1, 4, dtype=torch.long, device=device)

    if control:
        example_control = torch.zeros(1, 4, 3, dtype=torch.float, device=device)
        wrapper = TorchScriptGPT2ContWrapper(model)
        wrapper.eval()
        print(f"Tracing control model with {num_layers} layers...")
        traced = torch.jit.trace(
            wrapper, [example_input, example_control, empty_pkv],
            strict=False, check_trace=False,
        )
    else:
        wrapper = GPT2ManualWrapper(model)
        wrapper.eval()
        print(f"Tracing with {num_layers} layers...")
        traced = torch.jit.trace(
            wrapper, [example_input, empty_pkv],
            strict=False, check_trace=False,
        )

    device_str = "cuda" if device.type == "cuda" else "cpu"
    metadata = {
        "encoder": encoderX,
        "num_heads": int(num_heads),
        "num_hidden": int(num_hidden),
        "num_layers": int(num_layers),
        "model_dim": -1,
        "new_state": True,
        "traced_device": device_str,
    }
    print("Metadata:", metadata)

    extra_files = {"metadata.json": json.dumps(metadata)}
    torch.jit.save(traced, path, _extra_files=extra_files)
    print(f"Saved TorchScript model to {path} (traced on {device_str})")


def inject_metadata(path, metadata_path, encoder, new_state):
    model = torch.jit.load(path)
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    metadata["encoder"] = encoder
    metadata["new_state"] = new_state
    extra_files = torch._C.ExtraFilesMap()
    extra_files["metadata.json"] = json.dumps(metadata)
    out_path = os.path.splitext(path)[0] + "_WMETA.pt"
    torch.jit.save(model, out_path, _extra_files=extra_files)
    print(f"Saved with metadata to {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Convert HF checkpoint to TorchScript")
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--metadata_path", type=str, default="")
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--encoder", type=str, default="NONE")
    parser.add_argument("--init", action="store_true", help="Create from scratch (random weights)")
    parser.add_argument("--inject", action="store_true", help="Inject metadata into existing .pt")
    parser.add_argument("--new_state", action="store_true")
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--prune", action="store_true")
    parser.add_argument("--control", action="store_true", help="Use control-embedding model variant")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                        help="Device to trace on (baked into TorchScript graph)")

    args = parser.parse_args()

    if args.inject:
        assert args.metadata_path, "--metadata_path required with --inject"
        inject_metadata(args.ckpt_path, args.metadata_path, args.encoder, args.new_state)
        return

    assert args.output, "--output required"

    if args.init:
        encoder_mode = midigpt.getEncoderType(args.encoder)
        assert encoder_mode is not midigpt.ENCODER_TYPE.NO_ENCODER
        encoder = midigpt.getEncoder(encoder_mode)
        vocab_size = encoder.vocab_size()

        if args.control:
            assert GPT2LMHeadModelContConfig is not None, "custom_models not found"
            config = GPT2LMHeadModelContConfig().from_json_file(args.config)
            config.n_control_dim = encoder.config.embed_dim
            model = GPT2LMHeadModelCont(config)
        else:
            config = GPT2Config().from_json_file(args.config)
            config.vocab_size = vocab_size
            model = GPT2LMHeadModel(config)
    else:
        if args.control:
            assert GPT2LMHeadModelCont is not None, "custom_models not found"
            model = GPT2LMHeadModelCont.from_pretrained(args.ckpt_path)
        else:
            model = GPT2LMHeadModel.from_pretrained(args.ckpt_path)

    convert(
        model, args.output,
        quantize=args.quantize, prune=args.prune, force=args.force,
        control=args.control, ckpt_path=args.ckpt_path, encoderX=args.encoder,
        device=args.device,
    )


if __name__ == "__main__":
    main()
