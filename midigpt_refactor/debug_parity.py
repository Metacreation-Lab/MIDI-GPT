import pytest
import sys
from pathlib import Path
import midigpt_refactor._core as _core
from midigpt_refactor.tokenizer.tokenizer import Tokenizer
from midigpt_refactor.attributes import AttributeAnalyzer
import midigpt

def test_files():
    orig_enc = midigpt.ElVelocityDurationPolyphonyYellowEncoder()
    config_text = Path("../models/yellow_config.json").read_text()
    
    analyzer = AttributeAnalyzer.from_config(cfg)
    tokenizer = Tokenizer(_core.EncoderConfig.from_json(config_text), analyzer)
    
    midi_dir = Path("tests/comparison/midi")
    reader = _core.MidiReader(12)
    
    for mp in sorted(midi_dir.glob("*.mid")):
        print(f"Testing {mp.name}...")
        sys.stdout.flush()
        
        try:
            pj = orig_enc.midi_to_json(str(mp))
            orig = orig_enc.json_to_tokens(pj)
            print(f"  Orig: OK")
        except Exception as e:
            orig = None
            print(f"  Orig: Skipped ({e})")
            continue
            
        score = reader.read(str(mp))
        ref = tokenizer.encode(score, compute_attributes=True)
        print(f"  Ref: OK")

if __name__ == "__main__":
    test_files()
