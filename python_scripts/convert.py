"""Convert a trained HuggingFace GPT-2 checkpoint to TorchScript (.pt).

Embeds model metadata (encoder type, architecture) into the .pt file so the
C++ inference engine can load it without separate config files.

Compatible with transformers v4.x and v5.x (handles DynamicCache → tuples).

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
# TorchScript wrapper — bridges DynamicCache ↔ tuple interface
# ---------------------------------------------------------------------------

def _cache_to_tuples(cache):
    """Convert any transformers cache object to tuple-of-tuples of tensors."""
    if isinstance(cache, tuple):
        return cache
    # Transformers v5.2+: DynamicCache with .layers list of DynamicLayer
    if hasattr(cache, "layers"):
        return tuple((l.keys, l.values) for l in cache.layers)
    # Transformers v4.x-v5.1: DynamicCache with .key_cache/.value_cache lists
    if hasattr(cache, "key_cache"):
        return tuple(
            (k, v) for k, v in zip(cache.key_cache, cache.value_cache)
        )
    if hasattr(cache, "to_legacy_cache"):
        return cache.to_legacy_cache()
    return cache


def _tuples_to_cache(past_kv_tuples):
    """Convert tuple-of-tuples → DynamicCache for model input.

    Directly sets keys/values on DynamicLayer objects instead of using
    ``cache.update()`` which does ``torch.cat`` with an empty CPU tensor —
    that cat gets baked into TorchScript as a CPU constant and breaks when
    the model runs on GPU at inference time.
    """
    from transformers.cache_utils import DynamicCache, DynamicLayer
    cache = DynamicCache()
    cache.layers = []
    for key, value in past_kv_tuples:
        layer = DynamicLayer()
        layer.keys = key
        layer.values = value
        cache.layers.append(layer)
    return cache


class TorchScriptGPT2Wrapper(nn.Module):
    """Wraps GPT2LMHeadModel so forward() uses only tensors/tuples.

    Transformers v5+ uses DynamicCache internally, which TorchScript cannot
    handle. This wrapper converts between tuple-based past_key_values (what
    the C++ inference engine expects) and whatever the model uses internally.
    """

    def __init__(self, model: GPT2LMHeadModel):
        super().__init__()
        self.model = model
        self.model.config.use_cache = True
        self.model.config.torchscript = True

    def forward(self, input_ids, past_key_values):
        if past_key_values is not None and len(past_key_values) > 0:
            cache_input = _tuples_to_cache(past_key_values)
        else:
            cache_input = None

        outputs = self.model(input_ids=input_ids, past_key_values=cache_input)
        return outputs[0], _cache_to_tuples(outputs[1])


class TorchScriptGPT2ContWrapper(nn.Module):
    """Same as above but for the control-embedding variant."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.config.use_cache = True
        self.model.config.torchscript = True

    def forward(self, input_ids, control_ids, past_key_values):
        if past_key_values is not None and len(past_key_values) > 0:
            cache_input = _tuples_to_cache(past_key_values)
        else:
            cache_input = None

        outputs = self.model(
            input_ids=input_ids, control_ids=control_ids,
            past_key_values=cache_input,
        )
        return outputs[0], _cache_to_tuples(outputs[1])


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

    example_input = torch.zeros(1, 300, dtype=torch.long, device=device)

    if control:
        example_control = torch.zeros(1, 300, 3, dtype=torch.float, device=device)
        wrapper = TorchScriptGPT2ContWrapper(model)
        wrapper.eval()
        with torch.no_grad():
            logits, pkv = wrapper(example_input, example_control, ())
        print(f"Tracing with {len(pkv)} layers...")
        traced = torch.jit.trace(
            wrapper, [example_input, example_control, pkv],
            strict=False, check_trace=False,
        )
    else:
        wrapper = TorchScriptGPT2Wrapper(model)
        wrapper.eval()
        with torch.no_grad():
            logits, pkv = wrapper(example_input, ())
        print(f"Tracing with {len(pkv)} layers...")
        traced = torch.jit.trace(
            wrapper, [example_input, pkv],
            strict=False, check_trace=False,
        )

    num_layers = len(pkv)
    _, num_heads, _, num_hidden = pkv[0][0].shape

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
