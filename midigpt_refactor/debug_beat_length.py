import midigpt_refactor._core as _core
from midigpt_refactor._converters import from_cpp

def analyze_score_object(score, name):
    print(f"Analysis of Score Object '{name}':")
    print(f"Resolution: {score.resolution}")
    print(f"Tracks: {len(score.tracks)}")
    for i, track in enumerate(score.tracks):
        print(f"Track {i}: {len(track.bars)} bars")
        for b_idx, bar in enumerate(track.bars):
            print(f"  Bar {b_idx}: {len(bar.notes)} notes, beat_length={bar.beat_length}")

if __name__ == "__main__":
    # Test on a generated score before saving
    import torch
    from midigpt_refactor.tokenizer.tokenizer import Tokenizer
    from midigpt_refactor.inference.engine import InferenceEngine
    from midigpt_refactor.inference.config import GenerationRequest, TrackPrompt, SamplingConfig
    from midigpt_refactor import _core
    
    config_path = "../models/yellow_config.json"
    model_path = "../models/yellow.pt"
    midi_path = "tests/midi/piano_bell.mid"
    
    with open(config_path, "r") as f:
        config = _core.EncoderConfig.from_json(f.read())
    model = torch.jit.load(model_path, map_location="cpu")
    tokenizer = Tokenizer(config)
    engine = InferenceEngine(model, tokenizer, None)
    
    score = from_cpp(_core.MidiReader(config.resolution).read(midi_path))
    analyze_score_object(score, "Prompt Score")
    
    req = GenerationRequest(
        tracks=[TrackPrompt(id=0, bars=[0, 3], autoregressive=False)],
        config=SamplingConfig(max_attempts=1, silence_check=False)
    )
    
    with engine.session(score, req) as session:
        result_score = session.run()
    
    analyze_score_object(result_score, "Result Score")
