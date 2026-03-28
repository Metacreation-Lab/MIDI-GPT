"""Test generation with SteinbergWPCSEncoder and a TorchScript checkpoint.

Requires a build WITH LibTorch (no MIDIGPT_NO_TORCH).
Run on a GPU node or a build with torch available.

Usage:
    python test_steinberg_inference.py \
        --ckpt /path/to/checkpoint.pt \
        --midi tests/midi_files/singletrack/Maestro_1.mid \
        --out /scratch/$USER/gen_output.mid
"""

import argparse
import json
import os
import sys

sys.path.append(
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "build")
)
import midigpt


def main():
    parser = argparse.ArgumentParser(description="SteinbergWPCSEncoder inference test")
    parser.add_argument("--ckpt", type=str, required=True, help="TorchScript checkpoint path")
    parser.add_argument("--midi", type=str, required=True, help="Input MIDI file")
    parser.add_argument("--out", type=str, default="", help="Output MIDI path")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_steps", type=int, default=0)
    args = parser.parse_args()

    if not os.path.isfile(args.ckpt):
        print(f"ERROR: Checkpoint not found: {args.ckpt}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(args.midi):
        print(f"ERROR: MIDI file not found: {args.midi}", file=sys.stderr)
        sys.exit(1)

    out_path = args.out or os.path.join(
        os.path.dirname(args.midi), "steinberg_gen_output.mid"
    )

    # Use SteinbergWPCSEncoder (resolution=24, no microtiming, no velocity levels)
    enc = midigpt.SteinbergWPCSEncoder()
    print(f"Encoder: SteinbergWPCSEncoder")
    print(f"  vocab_size: {enc.vocab_size()}")
    print(f"  resolution: {enc.config.resolution}")
    print(f"  attribute_controls: {enc.get_attribute_control_types()}")

    # Parse MIDI
    piece_json = enc.midi_to_json(args.midi)
    piece = json.loads(piece_json)

    n_tracks = len(piece.get("tracks", []))
    if n_tracks == 0:
        print("ERROR: No tracks in MIDI file", file=sys.stderr)
        sys.exit(1)

    first_track = piece["tracks"][0]
    n_bars = len(first_track.get("bars", []))
    print(f"\nInput: {args.midi}")
    print(f"  tracks: {n_tracks}, bars: {n_bars}")

    if n_bars < 2:
        print("ERROR: Need at least 2 bars", file=sys.stderr)
        sys.exit(1)

    # Encode and check tokens
    tokens = enc.json_to_tokens(piece_json)
    print(f"  tokens: {len(tokens)}")
    assert all(0 <= t < enc.vocab_size() for t in tokens), "Token out of range!"

    # Build status: generate bars 1..(n_bars-2), condition on first and last
    selected = [False] + [True] * (n_bars - 2) + [False]
    if n_bars <= 2:
        selected = [True] * n_bars

    status = {
        "tracks": [
            {
                "track_id": 0,
                "track_type": first_track.get("trackType", 10),
                "temperature": args.temperature,
                "ignore": False,
                "selected_bars": selected,
                "autoregressive": False,
                "polyphony_hard_limit": 0,
            }
        ]
    }

    # Mark other tracks as ignored/conditioned
    for i in range(1, n_tracks):
        t = piece["tracks"][i]
        t_bars = len(t.get("bars", []))
        status["tracks"].append(
            {
                "track_id": i,
                "track_type": t.get("trackType", 10),
                "temperature": args.temperature,
                "ignore": False,
                "selected_bars": [False] * t_bars,
                "autoregressive": False,
            }
        )

    param = {
        "tracks_per_step": 1,
        "bars_per_step": min(2, n_bars),
        "model_dim": min(4, n_bars),
        "percentage": 100,
        "batch_size": 1,
        "temperature": args.temperature,
        "max_steps": args.max_steps,
        "polyphony_hard_limit": 0,
        "shuffle": False,
        "verbose": True,
        "ckpt": args.ckpt,
        "sampling_seed": args.seed,
        "mask_top_k": 0,
    }

    print(f"\nGenerating (seed={args.seed}, temp={args.temperature})...")
    callbacks = midigpt.CallbackManager()
    result = midigpt.sample_multi_step(
        json.dumps(piece), json.dumps(status), json.dumps(param), 3, callbacks
    )
    result_json = result[0]
    result_piece = json.loads(result_json)

    # Decode back to MIDI
    enc.json_to_midi(result_json, out_path)
    print(f"\nOutput written to: {out_path}")

    # Quick sanity check: re-encode the output
    out_tokens = enc.json_to_tokens(result_json)
    print(f"  output tokens: {len(out_tokens)}")
    assert all(0 <= t < enc.vocab_size() for t in out_tokens), "Output token out of range!"

    # Compute attribute controls on the output
    updated_json = midigpt.compute_all_attribute_controls(result_json)
    updated = json.loads(updated_json)
    for i, track in enumerate(updated.get("tracks", [])):
        feats = track.get("internalFeatures", [{}])
        if feats:
            print(f"\n  Track {i} features: {json.dumps(feats[0], indent=2)}")

    print("\nInference test PASSED")


if __name__ == "__main__":
    main()
