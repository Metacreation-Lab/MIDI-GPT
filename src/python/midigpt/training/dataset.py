from __future__ import annotations
import copy
import json
import random
import pyarrow as pa
import pyarrow.parquet as pq

import midigpt._core as _core
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
    """Training dataset with encoder-aware windowing, infill, and bar masking.

    Per-sample pipeline:
      1. Pitch/velocity augmentation on the full score (once).
      2. Independently roll two decisions:
           - is_infill  (infill_probability): encode with FillIn tokens.
           - do_mask    (mask_bar_config.apply_probability): apply MASK_BAR.
         These are independent. MaskBar is given the infill target bars so it
         never marks an infill target as MASK_BAR (mutually exclusive states).
      3. Retry loop over (n_bars, n_tracks) pairs:
           - Infill overflow → step n_tracks down, then n_bars down; never
             clip FillIn tokens.
           - AR overflow → random-window clip (legacy behaviour).
      4. Fallback: smallest window, 1 track, AR clip.

    max_seq_len acts as a dataset-side cap and must not exceed the model's
    positional budget (model.max_context()). Pass 0 to use the encoder's
    largest num_bars_map entry as an implicit limit.
    """

    def __init__(
        self,
        parquet_path: str,
        tokenizer: Tokenizer,
        # Infill training
        infill_probability: float = 0.75,
        infill_bar_fraction: float = 0.5,
        # Bar masking (independent of infill)
        mask_bar_config: MaskBarConfig | None = None,
        # Sequence budget
        max_seq_len: int = 2048,
        # Window / track sampling
        max_tracks: int = 12,
        min_tracks: int = 1,
        min_fill_ratio: float = 0.75,
    ):
        try:
            import datasets as hf
        except ImportError:
            raise ImportError("pip install midigpt[train]")
        self._data           = hf.load_dataset("parquet", data_files=parquet_path, split="train")
        self._tokenizer      = tokenizer
        self._infill_prob    = infill_probability
        self._infill_frac    = infill_bar_fraction
        self._mask_cfg       = mask_bar_config
        self._max_seq_len    = max_seq_len
        self._max_tracks     = max_tracks
        self._min_tracks     = min_tracks
        self._min_fill_ratio = min_fill_ratio

        cfg_dict = json.loads(tokenizer._vocab.config().to_json())
        raw_map  = cfg_dict.get("num_bars_map") or [4]
        self._num_bars_choices: list[int] = sorted(
            set(int(x) for x in raw_map), reverse=True
        )

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> dict:
        score = Score.from_dict(self._data[idx])
        score = copy.deepcopy(score)

        score = _TRANSPOSE(score)
        score = _VELOCITY(score)

        # Roll both decisions independently.
        is_infill = random.random() < self._infill_prob
        do_mask   = self._mask_cfg is not None  # gate is inside MaskBar itself

        initial   = random.choice(self._num_bars_choices)
        fallbacks = sorted([n for n in self._num_bars_choices if n < initial], reverse=True)

        for n_bars in [initial] + fallbacks:
            for n_tracks in range(self._max_tracks, self._min_tracks - 1, -1):
                window = select_window(score, n_bars, n_tracks, self._min_fill_ratio)
                if window is None:
                    continue

                # Sample a random per-cell probability uniformly in
                # [0, infill_bar_fraction], then independently apply it to
                # every (track, bar) cell. This varies density across samples
                # so the model trains on the full spectrum (few to many infill
                # targets). If no cell is selected, force one at random.
                infill_bars: set[tuple[int, int]] = set()
                if is_infill:
                    p = random.uniform(0.0, self._infill_frac)
                    infill_bars = {
                        (t_idx, b_idx)
                        for t_idx in range(len(window.tracks))
                        for b_idx in range(len(window.tracks[t_idx].bars))
                        if random.random() < p
                    }
                    if not infill_bars:
                        t = random.randrange(len(window.tracks))
                        b = random.randrange(len(window.tracks[t].bars))
                        infill_bars = {(t, b)}

                if do_mask:
                    window = MaskBar(self._mask_cfg, infill_bars=infill_bars)(window)

                encode_opts = _core.EncodeOptions()
                if is_infill and infill_bars:
                    encode_opts.multi_fill = infill_bars
                tokens = self._tokenizer.encode(window, opts=encode_opts)

                if is_infill and len(tokens) > self._max_seq_len:
                    # Never clip FillIn tokens — retry with fewer resources.
                    continue

                if len(tokens) > self._max_seq_len:
                    offset = random.randint(0, len(tokens) - self._max_seq_len)
                    tokens = tokens[offset : offset + self._max_seq_len]

                return {"input_ids": tokens, "labels": tokens}

        # Fallback: smallest window, 1 track, no infill, hard clip.
        window = select_window(
            score, self._num_bars_choices[-1], self._min_tracks, self._min_fill_ratio
        )
        tokens = self._tokenizer.encode(window if window else score)[: self._max_seq_len]
        return {"input_ids": tokens, "labels": tokens}
