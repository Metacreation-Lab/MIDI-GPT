from __future__ import annotations
import copy
import json
import random
import pyarrow as pa
import pyarrow.parquet as pq

from midigpt._types import Score
from midigpt.tokenizer.tokenizer import Tokenizer
from midigpt.augmentation.transpose import Transpose
from midigpt.augmentation.velocity import VelocityScale
from midigpt.augmentation.score_window import select_window
from midigpt.augmentation.mask_bar import MaskBar, MaskBarConfig


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
              splits: dict[str, float] | None = None):
        if splits is None:
            splits = {"train": 0.9, "valid": 0.05, "test": 0.05}
        scores = []
        for path in midi_paths:
            try:
                score = Score.from_midi(path)
                scores.append(score.to_dict())
            except Exception:
                pass
        table = pa.Table.from_pylist(scores, schema=_SCORE_SCHEMA)
        pq.write_table(table, output_path)


# ---------------------------------------------------------------------------
# Static augments applied once per score before windowing
# ---------------------------------------------------------------------------
_TRANSPOSE = Transpose(range(-6, 7))
_VELOCITY  = VelocityScale((0.8, 1.2))


class MidiGPTDataset:
    """Training dataset with encoder-aware windowing and infill sampling.

    Per-sample construction mirrors the legacy pipeline:
      1. Apply pitch/velocity augmentation to the full score.
      2. Roll infill vs. autoregressive (infill_probability, default 0.75).
      3. Pick n_bars randomly from `num_bars_map` in the encoder config.
      4. Try selecting n_tracks from max_tracks down to min_tracks:
           - Encode the windowed (and optionally masked) score.
           - If it fits in max_seq_len: done.
           - AR overflow: random-window clip (legacy behaviour).
           - Infill overflow: retry with one fewer track, then step down to
             the next smaller value in num_bars_map; never clip FillIn tokens.
      5. Fallback (all combinations exhausted): AR clip on smallest window.
    """

    def __init__(
        self,
        parquet_path: str,
        tokenizer: Tokenizer,
        mask_bar_config: MaskBarConfig | None = None,
        max_seq_len: int = 2048,
        max_tracks: int = 12,
        min_tracks: int = 1,
        min_fill_ratio: float = 0.75,
        infill_probability: float = 0.75,
    ):
        try:
            import datasets as hf
        except ImportError:
            raise ImportError("pip install midigpt[train]")
        self._data           = hf.load_dataset("parquet", data_files=parquet_path, split="train")
        self._tokenizer      = tokenizer
        self._mask_cfg       = mask_bar_config
        self._max_seq_len    = max_seq_len
        self._max_tracks     = max_tracks
        self._min_tracks     = min_tracks
        self._min_fill_ratio = min_fill_ratio
        self._infill_prob    = infill_probability if mask_bar_config is not None else 0.0

        # Read num_bars_map from encoder config; fall back to [4] if absent.
        cfg_dict = json.loads(tokenizer._vocab.config().to_json())
        raw_map = cfg_dict.get("num_bars_map") or [4]
        # Sorted descending so retry always steps to smaller values.
        self._num_bars_choices: list[int] = sorted(set(int(x) for x in raw_map), reverse=True)

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> dict:
        score = Score.from_dict(self._data[idx])
        score = copy.deepcopy(score)

        # Pitch and velocity augment the full score once.
        score = _TRANSPOSE(score)
        score = _VELOCITY(score)

        is_infill = self._mask_cfg is not None and random.random() < self._infill_prob

        # Build the retry order: random first pick, then descending fallbacks.
        initial = random.choice(self._num_bars_choices)
        fallbacks = sorted([n for n in self._num_bars_choices if n < initial], reverse=True)
        n_bars_order = [initial] + fallbacks

        for n_bars in n_bars_order:
            for n_tracks in range(self._max_tracks, self._min_tracks - 1, -1):
                window = select_window(score, n_bars, n_tracks, self._min_fill_ratio)
                if window is None:
                    continue

                if is_infill:
                    masked = MaskBar(self._mask_cfg)(copy.deepcopy(window))
                    tokens = self._tokenizer.encode(masked)
                    if len(tokens) <= self._max_seq_len:
                        return {"input_ids": tokens, "labels": tokens}
                    # Too long — try fewer tracks / smaller n_bars; never clip.
                    continue
                else:
                    tokens = self._tokenizer.encode(window)
                    if len(tokens) > self._max_seq_len:
                        # AR: random-window clip preserves training diversity.
                        offset = random.randint(0, len(tokens) - self._max_seq_len)
                        tokens = tokens[offset : offset + self._max_seq_len]
                    return {"input_ids": tokens, "labels": tokens}

        # All combinations exhausted (extremely rare: score has very few bars).
        # Fall back: smallest window, 1 track, AR clip.
        window = select_window(score, self._num_bars_choices[-1], self._min_tracks, self._min_fill_ratio)
        if window is None:
            tokens = self._tokenizer.encode(score)[: self._max_seq_len]
        else:
            tokens = self._tokenizer.encode(window)[: self._max_seq_len]
        return {"input_ids": tokens, "labels": tokens}
