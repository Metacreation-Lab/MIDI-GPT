import pytest
from midigpt._types import Score, Track, Bar, Note
from midigpt.tokenizer import Tokenizer
import midigpt._core as _core
import json
import tempfile
import pathlib

MINIMAL_CONFIG_JSON = json.dumps({
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
    ]
})

def test_tokenizer_encode_decode():
    config = _core.EncoderConfig.from_json(MINIMAL_CONFIG_JSON)
    tokenizer = Tokenizer(config)
    
    n = Note(pitch=60, velocity=100, onset_ticks=0, duration_ticks=480)
    b = Bar(notes=[n], ts_numerator=4, ts_denominator=4)
    t = Track(bars=[b], instrument=0, track_type="melodic")
    s = Score(tracks=[t])
    
    tokens = tokenizer.encode(s)
    assert len(tokens) > 0
    
    # decode back
    s2 = tokenizer.decode(tokens)
    assert len(s2.tracks) == 1
    assert len(s2.tracks[0].bars) == 1
    assert s2.tracks[0].bars[0].notes[0].pitch == 60
