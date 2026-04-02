"""test_hf_checkpoint.py

Test a raw HuggingFace checkpoint directly (no TorchScript conversion) to
verify the underlying model weights are good before blaming conversion.

Runs the same autoregressive loop as simulate_generation.py but using
GPT2LMHeadModel.generate-style step-by-step with real past_key_values.
"""

import argparse, json, os, sys
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "build"))
import midigpt


NOTE_NAMES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
def note_name(p): return f"{NOTE_NAMES[p%12]}{p//12-1}"


def sample_token(logits, temperature=1.0, top_k=50, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    logits = logits / temperature
    if top_k > 0:
        vals, _ = torch.topk(logits, top_k)
        logits[logits < vals[-1]] = -float("inf")
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1).item()


def generate_hf(model, context_tokens, n_new=40, temperature=1.0, seed=42):
    """Proper autoregressive generation using HF model with real KV cache."""
    torch.manual_seed(seed)
    model.eval()

    ids = torch.tensor([context_tokens], dtype=torch.long)
    generated = []
    past_key_values = None

    with torch.no_grad():
        # Prefill
        out = model(input_ids=ids, past_key_values=None, use_cache=True)
        logits = out.logits        # [1, seq_len, vocab]
        past_key_values = out.past_key_values

        for step in range(n_new):
            next_tok = sample_token(logits[0, -1], temperature=temperature)
            generated.append(next_tok)

            new_id = torch.tensor([[next_tok]], dtype=torch.long)
            out = model(input_ids=new_id, past_key_values=past_key_values, use_cache=True)
            logits = out.logits
            past_key_values = out.past_key_values

    return generated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="HuggingFace checkpoint dir")
    parser.add_argument("--midi", required=True)
    parser.add_argument("--n_new", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.9)
    args = parser.parse_args()

    enc_name = "EL_VELOCITY_DURATION_POLYPHONY_YELLOW_ENCODER"
    enc = midigpt.getEncoder(midigpt.getEncoderType(enc_name))
    context_tokens = enc.midi_to_tokens(args.midi)
    print(f"Context: {len(context_tokens)} tokens")

    print(f"Loading HF checkpoint: {args.ckpt}")
    model = GPT2LMHeadModel.from_pretrained(args.ckpt)
    model.eval()
    print(f"  vocab_size={model.config.vocab_size}  n_layer={model.config.n_layer}")

    print(f"Generating {args.n_new} tokens (temp={args.temperature})...")
    gen = generate_hf(model, context_tokens, args.n_new, args.temperature)

    print(f"\nGenerated tokens : {gen}")

    full_tokens = list(context_tokens) + gen
    try:
        midi_json = enc.tokens_to_json(full_tokens)
        data = json.loads(midi_json)
        note_ons = [e for e in data.get("events", []) if e.get("velocity", 0) > 0]
        context_pitches = set(e["pitch"] for e in note_ons[:len(context_tokens)])
        all_pitches = [e["pitch"] for e in note_ons]
        print(f"All pitches      : {[note_name(p) for p in all_pitches]}")

        # Separate context vs generated
        context_note_count = sum(1 for e in note_ons
                                  if e.get("pitch") in {60,64,67,65,69,72,67,71,74})
        print(f"(first {context_note_count} are context chord notes C-F-G-C)")
        gen_notes = [note_name(e["pitch"]) for e in note_ons[12:]]  # skip 4 chords × 3
        print(f"Generated notes  : {gen_notes}")
    except Exception as e:
        print(f"Decode failed: {e}")


if __name__ == "__main__":
    main()
