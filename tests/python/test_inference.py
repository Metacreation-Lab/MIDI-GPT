import pytest
import json
from midigpt._types import Score, Track, Bar, Note
from midigpt.tokenizer import Tokenizer
from midigpt.inference import InferenceEngine, SamplingConfig, TrackPrompt, GenerationRequest
from midigpt.attributes.base import AttributeAnalyzer
import midigpt._core as _core

torch = pytest.importorskip("torch")


class StubModel:
    """Minimal TorchScript-like stub: returns uniform logits over the vocab."""
    def __init__(self, vocab_size: int = 512):
        self._vocab = vocab_size

    def __call__(self, input_ids, past_kv=None):
        batch, seq = input_ids.shape
        logits = torch.zeros(batch, seq, self._vocab)
        return logits

    def eval(self):
        pass

MINIMAL_CONFIG_JSON = json.dumps({
    "supports_infill": True,
    "token_domains": [
        {"type": "PieceStart", "domain_size": 1},
        {"type": "PieceEnd", "domain_size": 1},
        {"type": "Track", "domain_size": 2},
        {"type": "TrackEnd", "domain_size": 1},
        {"type": "Bar", "domain_size": 1},
        {"type": "BarEnd", "domain_size": 1},
        {"type": "Instrument", "domain_size": 128},
        {"type": "TimeSig", "domain_size": 32},
        {"type": "NoteOnset", "domain_size": 128},
        {"type": "NoteDuration", "domain_size": 128},
        {"type": "VelocityLevel", "domain_size": 128},
        {"type": "DeltaDirection", "domain_size": 2},
        {"type": "Delta", "domain_size": 128},
        {"type": "NoteDensity", "domain_size": 128},
        {"type": "OnsetPolyphony", "domain_size": 128},
        {"type": "PitchRange", "domain_size": 128},
        {"type": "PitchClassSet", "domain_size": 128}
    ],
    "attribute_controls": [
        {"name": "note_density"},
        {"name": "onset_polyphony"},
        {"name": "pitch_range"},
        {"name": "key_signature"},
        {"name": "note_duration_dist"},
        {"name": "silence_proportion"},
        {"name": "pitch_class_set"}
    ]
})

def test_inference_session():
    config = _core.EncoderConfig.from_json(MINIMAL_CONFIG_JSON)
    tokenizer = Tokenizer(config)
    engine = InferenceEngine(StubModel(tokenizer.vocab_size()), tokenizer, AttributeAnalyzer.from_config(config))

    bars = [
        Bar(notes=[Note(pitch=60, velocity=100, onset_ticks=0, duration_ticks=480)],
            ts_numerator=4, ts_denominator=4)
        for _ in range(4)
    ]
    t = Track(bars=bars, instrument=0, track_type="melodic")
    s = Score(tracks=[t])

    req = GenerationRequest(
        tracks=[TrackPrompt(id=0, bars=[0])],
        config=SamplingConfig(max_attempts=1, silence_check=False, novelty_check=False)
    )

    with engine.session(s, req) as session:
        s2 = session.run()
        assert isinstance(s2, Score)

def test_load_checkpoint_single_file_bundle(tmp_path):
    """Single-file .pt bundle {config, encoder_config, state_dict} loads end-to-end."""
    torch = pytest.importorskip("torch")
    from midigpt.inference.model import GPT2LMHeadModel, GPT2Config
    from midigpt.tokenizer.checkpoint import load_checkpoint

    enc_cfg_dict = json.loads(MINIMAL_CONFIG_JSON)
    enc_cfg = _core.EncoderConfig.from_json(MINIMAL_CONFIG_JSON)
    vocab_size = Tokenizer(enc_cfg).vocab_size()
    cfg = GPT2Config(vocab_size=vocab_size, n_positions=64,
                     n_embd=16, n_layer=2, n_head=2)
    model = GPT2LMHeadModel(cfg)
    model.encoder_config = enc_cfg_dict
    bundle_path = tmp_path / "tiny.pt"
    model.save_pretrained(str(bundle_path))

    bundle = load_checkpoint(str(bundle_path))
    assert bundle.model is not None
    assert bundle.model_path is None

    engine = InferenceEngine.from_checkpoint(str(bundle_path))

    bars = [Bar(notes=[Note(pitch=60, velocity=100, onset_ticks=0, duration_ticks=480)],
                ts_numerator=4, ts_denominator=4) for _ in range(4)]
    s = Score(tracks=[Track(bars=bars, instrument=0, track_type="melodic")])
    req = GenerationRequest(
        tracks=[TrackPrompt(id=0, bars=[2, 3])],
        config=SamplingConfig(max_attempts=1, silence_check=False, novelty_check=False),
    )
    with engine.session(s, req) as session:
        out = session.run()
        assert isinstance(out, Score)


def test_inference_attributes():
    config = _core.EncoderConfig.from_json(MINIMAL_CONFIG_JSON)
    tokenizer = Tokenizer(config)
    engine = InferenceEngine(StubModel(tokenizer.vocab_size()), tokenizer, AttributeAnalyzer.from_config(config))

    bars = [
        Bar(notes=[Note(pitch=60, velocity=100, onset_ticks=0, duration_ticks=480)],
            ts_numerator=4, ts_denominator=4)
        for _ in range(4)
    ]
    s = Score(tracks=[Track(bars=bars, instrument=0, track_type="melodic")])
    
    req = GenerationRequest(
        tracks=[TrackPrompt(
            id=0, 
            bars=[0],
            attributes={
                "note_density": 5,
                "onset_polyphony": 2,
                "pitch_range": 4,
                "key_signature": 3
            }
        )],
        config=SamplingConfig(max_attempts=1, silence_check=False, novelty_check=False)
    )
    
    # Run session - this will test _build_constraints mapping
    with engine.session(s, req) as session:
        s2 = session.run()
        assert isinstance(s2, Score)
