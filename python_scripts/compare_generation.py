"""compare_generation.py

Run sample_multi_step on two .pt models with the same MIDI input and compare
the generated token sequences / decoded note output side by side.

Usage:
    python compare_generation.py \
        --model_a /path/to/model.pt \
        --model_b /path/to/other.pt \
        --midi /path/to/test.mid
"""

import argparse
import json
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "build"))
import midigpt  # noqa: E402


NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

def midi_note_name(n):
    return f"{NOTE_NAMES[n % 12]}{n // 12 - 1}"


def run_inference(model_path, midi_path, bar_to_generate=2, temperature=1.0):
    encoder_mode_str = "EL_VELOCITY_DURATION_POLYPHONY_YELLOW_ENCODER"
    encoder_mode = midigpt.getEncoderType(encoder_mode_str)
    encoder = midigpt.getEncoder(encoder_mode)

    midi_json = encoder.midi_to_json(midi_path)
    midi_data = json.loads(midi_json)

    num_bars = len(midi_data["tracks"][0]["bars"])

    # Mark the last bar as selected for generation
    selected_bars = [False] * num_bars
    selected_bars[-1] = True

    status = json.dumps({
        "tracks": [{
            "track_id": 0,
            "temperature": temperature,
            "instrument": "acoustic_grand_piano",
            "density": 5,
            "track_type": 0,
            "ignore": False,
            "selected_bars": selected_bars,
            "min_polyphony_q": "POLYPHONY_ANY",
            "max_polyphony_q": "POLYPHONY_ANY",
            "autoregressive": False,
            "polyphony_hard_limit": 6,
        }]
    })

    param = json.dumps({
        "tracks_per_step": 1,
        "bars_per_step": 1,
        "model_dim": 4,
        "percentage": 100,
        "batch_size": 1,
        "temperature": temperature,
        "max_steps": 200,
        "polyphony_hard_limit": 6,
        "shuffle": False,
        "verbose": False,
        "ckpt": model_path,
        "sampling_seed": 42,
        "mask_top_k": 0,
    })

    callbacks = midigpt.CallbackManager()
    result = midigpt.sample_multi_step(midi_json, status, param, 1, callbacks)
    return json.loads(result[0])


def extract_notes(midi_data):
    """Extract all (pitch, velocity, bar_idx) from a MIDI JSON dict."""
    notes = []
    for track in midi_data.get("tracks", []):
        for bar_idx, bar in enumerate(track.get("bars", [])):
            for evt in bar.get("events", []):
                if isinstance(evt, dict) and evt.get("velocity", 0) > 0:
                    notes.append({
                        "bar": bar_idx,
                        "pitch": evt.get("pitch", -1),
                        "name": midi_note_name(evt["pitch"]) if evt.get("pitch") is not None else "?",
                        "velocity": evt.get("velocity"),
                        "onset": evt.get("onset"),
                    })
    return notes


def summarise_notes(notes, label):
    print(f"\n{'='*60}")
    print(f"  {label}  —  {len(notes)} notes")
    print(f"{'='*60}")
    if not notes:
        print("  (no notes found)")
        return
    pitches = [n["pitch"] for n in notes if n["pitch"] >= 0]
    if pitches:
        unique = sorted(set(midi_note_name(p) for p in pitches))
        print(f"  pitch range  : {midi_note_name(min(pitches))} – {midi_note_name(max(pitches))}")
        print(f"  unique notes : {unique}")
    # Print bar-by-bar
    bars = sorted(set(n["bar"] for n in notes))
    for b in bars:
        bar_notes = [n["name"] for n in notes if n["bar"] == b]
        print(f"  bar {b:2d}       : {bar_notes}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_a", required=True)
    parser.add_argument("--model_b", required=True)
    parser.add_argument("--midi", required=True)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--out_dir", default="/tmp")
    args = parser.parse_args()

    name_a = os.path.basename(args.model_a)
    name_b = os.path.basename(args.model_b)

    print(f"Input MIDI : {args.midi}")
    print(f"Model A    : {name_a}")
    print(f"Model B    : {name_b}")
    print(f"Temperature: {args.temperature}")

    # Show input notes
    import json
    enc = midigpt.getEncoder(midigpt.getEncoderType("EL_VELOCITY_DURATION_POLYPHONY_YELLOW_ENCODER"))
    input_data = json.loads(enc.midi_to_json(args.midi))
    input_notes = extract_notes(input_data)
    summarise_notes(input_notes, "INPUT")

    print(f"\nRunning inference with {name_a} ...")
    result_a = run_inference(args.model_a, args.midi, temperature=args.temperature)
    notes_a = extract_notes(result_a)
    summarise_notes(notes_a, f"OUTPUT A — {name_a}")

    # Save output A
    out_a = os.path.join(args.out_dir, "gen_model_a.mid")
    enc.json_to_midi(json.dumps(result_a), out_a)
    print(f"  saved → {out_a}")

    print(f"\nRunning inference with {name_b} ...")
    result_b = run_inference(args.model_b, args.midi, temperature=args.temperature)
    notes_b = extract_notes(result_b)
    summarise_notes(notes_b, f"OUTPUT B — {name_b}")

    out_b = os.path.join(args.out_dir, "gen_model_b.mid")
    enc.json_to_midi(json.dumps(result_b), out_b)
    print(f"  saved → {out_b}")

    # Quick overlap analysis
    print(f"\n{'='*60}")
    print("  OVERLAP ANALYSIS (generated bar)")
    print(f"{'='*60}")
    gen_bar = max(n["bar"] for n in input_notes) if input_notes else 0
    gen_a = set(n["name"] for n in notes_a if n["bar"] >= gen_bar)
    gen_b = set(n["name"] for n in notes_b if n["bar"] >= gen_bar)
    both = gen_a & gen_b
    print(f"  notes in A only : {sorted(gen_a - gen_b)}")
    print(f"  notes in B only : {sorted(gen_b - gen_a)}")
    print(f"  notes in both   : {sorted(both)}")


if __name__ == "__main__":
    main()
