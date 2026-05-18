import pyarrow as pa
import pyarrow.parquet as pq
from midigpt_refactor._types import Score
from midigpt_refactor.tokenizer.tokenizer import Tokenizer
from midigpt_refactor.augmentation.base import AugmentationPipeline
import math

_SCORE_SCHEMA = pa.schema([
    pa.field("resolution", pa.int32()),
    pa.field("tempo",      pa.int32()),
    pa.field("tracks", pa.list_(pa.struct([
        pa.field("instrument",  pa.int32()),
        pa.field("track_type",  pa.string()),
        pa.field("bars", pa.list_(pa.struct([
            pa.field("ts_numerator",   pa.int32()),
            pa.field("ts_denominator", pa.int32()),
            pa.field("future",         pa.bool_()),
            pa.field("notes", pa.list_(pa.struct([
                pa.field("pitch",          pa.int32()),
                pa.field("velocity",       pa.int32()),
                pa.field("onset_ticks",    pa.int32()),
                pa.field("duration_ticks", pa.int32()),
                pa.field("delta",          pa.int32()),
            ]))),
        ]))),
    ]))),
])

class DatasetBuilder:
    def build(self, midi_paths: list[str], output_path: str,
              splits: dict[str,float] = None):
        if splits is None:
            splits = {"train": 0.9, "valid": 0.05, "test": 0.05}
            
        scores = []
        for path in midi_paths:
            try:
                score = Score.from_midi(path)
                scores.append(score.to_dict())
            except Exception:
                pass
                
        # Write flat parquet file with all parsed scores
        table = pa.Table.from_pylist(scores, schema=_SCORE_SCHEMA)
        pq.write_table(table, output_path)

class MidiGPTDataset:
    def __init__(self, parquet_path: str, tokenizer: Tokenizer,
                 augmenter: AugmentationPipeline | None = None,
                 max_seq_len: int = 2048):
        try:
            import datasets as hf
        except ImportError:
            raise ImportError("pip install midigpt[train]")
        self._data        = hf.load_dataset("parquet", data_files=parquet_path, split="train")
        self._tokenizer   = tokenizer
        self._augmenter   = augmenter
        self._max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> dict:
        score = Score.from_dict(self._data[idx])
        if self._augmenter:
            score = self._augmenter(score)
        tokens = self._tokenizer.encode(score)[:self._max_seq_len]
        return {"input_ids": tokens, "labels": tokens}
