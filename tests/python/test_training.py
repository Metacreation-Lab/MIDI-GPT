import copy
import json
import os
import tempfile

import pytest

pytest.importorskip("pyarrow", reason="train extra not installed")
pytest.importorskip("torch", reason="inference extra not installed")

import pyarrow as pa
import pyarrow.parquet as pq
import torch

import midigpt._core as _core
from midigpt._types import Score, Track, Bar, Note
from midigpt.tokenizer import Tokenizer
from midigpt.training.dataset import DatasetBuilder, MidiGPTDataset, _SCORE_SCHEMA
from midigpt.training.collator import MidiGPTCollator

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

# Superset of MINIMAL_CONFIG_JSON that enables infill and bar-masking tokens.
INFILL_CONFIG_JSON = json.dumps({
    "supports_infill": True,
    "token_domains": [
        {"type": "PieceStart", "domain_size": 1},
        {"type": "PieceEnd", "domain_size": 1},
        {"type": "FillInPlaceholder", "domain_size": 1},
        {"type": "FillInStart", "domain_size": 1},
        {"type": "FillInEnd", "domain_size": 1},
        {"type": "MaskBar", "domain_size": 1},
        {"type": "Track", "domain_size": 2},
        {"type": "TrackEnd", "domain_size": 1},
        {"type": "Bar", "domain_size": 1},
        {"type": "BarEnd", "domain_size": 1},
        {"type": "Instrument", "domain_size": 128},
        {"type": "TimeSig", "domain_size": 32},
        {"type": "NoteOnset", "domain_size": 128},
        {"type": "NoteDuration", "domain_size": 128},
        {"type": "VelocityLevel", "domain_size": 128},
    ]
})

def _make_score(n_tracks: int = 2, n_bars: int = 4) -> Score:
    """Build a minimal score with non-empty bars."""
    notes = [Note(pitch=60 + i, velocity=80, onset_ticks=i * 120, duration_ticks=120)
             for i in range(4)]
    tracks = []
    for _ in range(n_tracks):
        bars = [copy.deepcopy(Bar(notes=notes, ts_numerator=4, ts_denominator=4))
                for _ in range(n_bars)]
        tracks.append(Track(bars=bars, instrument=0, track_type="melodic"))
    return Score(tracks=tracks)


def _write_parquet(scores: list, path: str) -> None:
    table = pa.Table.from_pylist([s.to_dict() for s in scores], schema=_SCORE_SCHEMA)
    pq.write_table(table, path)


def test_training_dataset():
    s = _make_score(n_tracks=1, n_bars=1)
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        path = f.name
    try:
        _write_parquet([s], path)
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


def test_training_batch_smoke():
    """Collate several batches and verify shapes, dtypes, and masking."""
    from midigpt.augmentation.mask_bar import MaskBarConfig, MaskMode

    n_scores, n_tracks, n_bars = 8, 2, 4
    scores = [_make_score(n_tracks=n_tracks, n_bars=n_bars) for _ in range(n_scores)]

    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        path = f.name
    try:
        _write_parquet(scores, path)

        config = _core.EncoderConfig.from_json(INFILL_CONFIG_JSON)
        tokenizer = Tokenizer(config)
        mask_cfg = MaskBarConfig(apply_probability=0.5, mode=MaskMode.MIXED)
        dataset = MidiGPTDataset(
            path, tokenizer=tokenizer,
            infill_probability=0.5,
            infill_bar_fraction=0.5,
            mask_bar_config=mask_cfg,
            max_seq_len=512,
            max_tracks=n_tracks,
            min_tracks=1,
        )
        assert len(dataset) == n_scores

        collator = MidiGPTCollator(max_seq_len=512)

        for batch_size in (1, 4):
            batch = [dataset[i % n_scores] for i in range(batch_size)]
            result = collator(batch)

            assert set(result.keys()) == {"input_ids", "attention_mask", "labels"}

            B, L = result["input_ids"].shape
            assert B == batch_size
            assert result["attention_mask"].shape == (B, L)
            assert result["labels"].shape == (B, L)

            assert result["input_ids"].dtype == torch.long
            assert result["attention_mask"].dtype == torch.long
            assert result["labels"].dtype == torch.long

            # Every sequence has at least one real token.
            assert (result["attention_mask"].sum(dim=1) > 0).all()
            # Padding positions carry label -100; real positions copy input_ids.
            real = result["attention_mask"].bool()
            assert (result["labels"][real] == result["input_ids"][real]).all()
            pad = ~real
            if pad.any():
                assert (result["labels"][pad] == -100).all()
    finally:
        os.remove(path)
