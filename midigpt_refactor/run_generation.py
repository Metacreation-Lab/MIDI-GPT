import argparse
import logging # Added import for logging
from pathlib import Path
from midigpt_refactor.inference.engine import InferenceEngine
from midigpt_refactor.inference.config import GenerationRequest, TrackPrompt, SamplingConfig
from midigpt_refactor._types import Score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to the model .pt file")
    parser.add_argument("--config", required=True, help="Path to the config.json file")
    parser.add_argument("--midi", required=True, help="Path to input midi prompt")
    parser.add_argument("--out", default="output.mid", help="Path to output midi file")
    parser.add_argument("--note_density", type=int, default=5)
    args = parser.parse_args()

    import torch
    from midigpt_refactor.tokenizer.tokenizer import Tokenizer
    from midigpt_refactor import _core
    from midigpt_refactor._converters import from_cpp # Added import

    _core.set_verbosity(_core.LogLevel.DEBUG)
    logging.basicConfig(level=logging.DEBUG)

    print(f"Loading config {args.config}...")
    with open(args.config, "r") as f:
        config_str = f.read()
    config = _core.EncoderConfig.from_json(config_str)

    print(f"Loading model {args.model}...")
    model = torch.jit.load(args.model, map_location="cpu")
    model.eval()

    tokenizer = Tokenizer(config)
    engine = InferenceEngine(model, tokenizer, None)

    # Use the config resolution when reading the MIDI
    score = from_cpp(_core.MidiReader(config.resolution).read(args.midi))

    req = GenerationRequest(
        tracks=[TrackPrompt(
            id=0,
            bars=[0, 1, 2, 3],  # Generate into the 4 newly added empty bars
            autoregressive=False
        )],
        config=SamplingConfig(
            max_attempts=10,
            temperature=1.0,
            silence_check=True,
            novelty_check=True
        )
    )

    print("Running generation...")
    # Trace the tokens if we could, but let's just run and then analyze result
    with engine.session(score, req) as session:
        result_score = session.run()

    print(f"Generation complete! Result score has {len(result_score.tracks)} tracks.")
    for i, t in enumerate(result_score.tracks):
        n_notes = sum(len(b.notes) for b in t.bars)
        print(f"  Track {i}: {len(t.bars)} bars, total notes in bars: {n_notes}")

    print(f"Saving to {args.out}")
    result_score.to_midi(args.out)

if __name__ == "__main__":
    main()
