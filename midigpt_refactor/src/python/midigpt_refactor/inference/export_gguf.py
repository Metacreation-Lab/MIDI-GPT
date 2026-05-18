"""Export the remapped GPT-2 checkpoint to GGUF for llama.cpp.

Writes a GGUF file with the GPT-2 architecture. The vocab is filled with
stringified integer IDs ("tok_0", "tok_1", ...) since our tokens are integers
already — at inference time we call `Llama.eval([token_ids...])` directly,
bypassing the BPE tokenizer.

Usage:
    .venv/bin/python3 midigpt_refactor/scripts/export_gguf.py \\
        --ckpt models/yellow_remapped.pt --out models/yellow.gguf
"""
import argparse
import numpy as np
import torch
import gguf
from midigpt_refactor.inference.model import GPT2LMHeadModel


def export(ckpt_path: str, out_path: str) -> None:
    model = GPT2LMHeadModel.from_torchscript(ckpt_path)
    model.eval()
    cfg = model.cfg
    sd = model.state_dict()

    writer = gguf.GGUFWriter(out_path, gguf.MODEL_ARCH_NAMES[gguf.MODEL_ARCH.GPT2])

    # ---- metadata ----
    writer.add_name("midigpt-yellow")
    writer.add_context_length(cfg.n_positions)
    writer.add_embedding_length(cfg.n_embd)
    writer.add_block_count(cfg.n_layer)
    writer.add_feed_forward_length(4 * cfg.n_embd)
    writer.add_head_count(cfg.n_head)
    writer.add_layer_norm_eps(1e-5)
    writer.add_file_type(gguf.LlamaFileType.ALL_F32)

    # ---- vocab (placeholder; we feed integer IDs at inference) ----
    tokens = [f"tok_{i}".encode("utf-8") for i in range(cfg.vocab_size)]
    scores = [0.0] * cfg.vocab_size
    toktypes = [gguf.TokenType.NORMAL] * cfg.vocab_size
    writer.add_tokenizer_model("gpt2")
    writer.add_tokenizer_pre("default")
    writer.add_token_list(tokens)
    writer.add_token_scores(scores)
    writer.add_token_types(toktypes)
    # GPT-2 tokenizer needs merges; provide a placeholder pair so the loader
    # accepts the file. We feed token IDs directly via Llama.eval([...]) and
    # never invoke BPE merging.
    writer.add_token_merges(["tok_0 tok_1".encode("utf-8")])

    # ---- tensors (GPT2 tensor names per llama.cpp's gguf-py mapping) ----
    # GGUF GPT-2 expects: token_embd, position_embd, output_norm, output,
    # blk.{i}.{attn_norm, attn_qkv, attn_output, ffn_norm, ffn_up, ffn_down}
    def t(name): return sd[name].detach().cpu().numpy().astype(np.float32)

    writer.add_tensor("token_embd.weight",   t("transformer.wte.weight"))
    writer.add_tensor("position_embd.weight", t("transformer.wpe.weight"))
    writer.add_tensor("output_norm.weight",  t("transformer.ln_f.weight"))
    writer.add_tensor("output_norm.bias",    t("transformer.ln_f.bias"))
    # lm_head shares weights with wte in many GPT-2 configs but here they are
    # separate after remapping. GGUF GPT-2 uses output.weight.
    writer.add_tensor("output.weight",       t("lm_head.weight"))

    for i in range(cfg.n_layer):
        p = f"transformer.h.{i}"
        # ln_1
        writer.add_tensor(f"blk.{i}.attn_norm.weight", t(f"{p}.ln_1.weight"))
        writer.add_tensor(f"blk.{i}.attn_norm.bias",   t(f"{p}.ln_1.bias"))
        # c_attn is HF Conv1D (nx, nf) -> need to transpose for GGUF/Linear
        writer.add_tensor(f"blk.{i}.attn_qkv.weight",  t(f"{p}.attn.c_attn.weight").T.copy())
        writer.add_tensor(f"blk.{i}.attn_qkv.bias",    t(f"{p}.attn.c_attn.bias"))
        writer.add_tensor(f"blk.{i}.attn_output.weight", t(f"{p}.attn.c_proj.weight").T.copy())
        writer.add_tensor(f"blk.{i}.attn_output.bias",   t(f"{p}.attn.c_proj.bias"))
        # ln_2
        writer.add_tensor(f"blk.{i}.ffn_norm.weight",  t(f"{p}.ln_2.weight"))
        writer.add_tensor(f"blk.{i}.ffn_norm.bias",    t(f"{p}.ln_2.bias"))
        # mlp.c_fc / c_proj
        writer.add_tensor(f"blk.{i}.ffn_up.weight",    t(f"{p}.mlp.c_fc.weight").T.copy())
        writer.add_tensor(f"blk.{i}.ffn_up.bias",      t(f"{p}.mlp.c_fc.bias"))
        writer.add_tensor(f"blk.{i}.ffn_down.weight",  t(f"{p}.mlp.c_proj.weight").T.copy())
        writer.add_tensor(f"blk.{i}.ffn_down.bias",    t(f"{p}.mlp.c_proj.bias"))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"GGUF → {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    export(args.ckpt, args.out)
