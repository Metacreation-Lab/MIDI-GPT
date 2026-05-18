import pytest
from midigpt._types import Score, Track, Bar, Note
from midigpt.tokenizer import Tokenizer
from midigpt.training.dataset import DatasetBuilder, MidiGPTDataset
import midigpt._core as _core
import json
import os
import tempfile
import pyarrow as pa
import pyarrow.parquet as pq

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

def test_training_dataset():
    # Write a dummy score manually to parquet to test reading
    n = Note(pitch=60, velocity=100, onset_ticks=0, duration_ticks=480)
    b = Bar(notes=[n], ts_numerator=4, ts_denominator=4)
    t = Track(bars=[b], instrument=0, track_type="melodic")
    s = Score(tracks=[t])
    
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        path = f.name
        
    try:
        builder = DatasetBuilder()
        # Instead of parsing MIDI, we'll write table manually for speed
        from midigpt.training.dataset import _SCORE_SCHEMA
        table = pa.Table.from_pylist([s.to_dict()], schema=_SCORE_SCHEMA)
        pq.write_table(table, path)
        
        config = _core.EncoderConfig.from_json(MINIMAL_CONFIG_JSON)
        tokenizer = Tokenizer(config)
        dataset = MidiGPTDataset(path, tokenizer=tokenizer)
        
        assert len(dataset) == 1
        item = dataset[0]
        assert "input_ids" in item
        assert "labels" in item
        assert len(item["input_ids"]) > 0
    finally:
        os.remove(path)
