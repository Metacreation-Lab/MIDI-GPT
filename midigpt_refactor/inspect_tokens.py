import argparse
import json
from midigpt.tokenizer.tokenizer import Tokenizer
from midigpt._types import Score, Bar, Note
from midigpt import _core

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--midi", required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config_str = f.read()
    config = _core.EncoderConfig.from_json(config_str)
    
    tokenizer = Tokenizer(config)
    score = Score.from_midi(args.midi)
    
    # Add 4 dummy bars
    for _ in range(4):
        score.tracks[0].bars.append(Bar())
    
    from midigpt._converters import to_cpp
    cpp_score = to_cpp(score)
    
    encoder = _core.Encoder(_core.Vocabulary(config))
    tokens = encoder.encode(cpp_score)
    
    print(f"Encoded {len(tokens)} tokens.")
    print("First 100 tokens (decoded):")
    for i, t in enumerate(tokens[:100]):
        tt, val = tokenizer.vocab.decode(t)
        print(f"  {i:3d}: {tokenizer._core_type_to_str(tt):20} val={val:3d} (token_id={t:4d})")

if __name__ == "__main__":
    main()
