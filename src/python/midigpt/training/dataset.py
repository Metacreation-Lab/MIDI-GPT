from __future__ import annotations
import copy
import hashlib
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
from midigpt.utils import cached_indices, file_cache_key


# ── dataset filter ────────────────────────────────────────────────────────────

def _metadata_valid_mask(parquet_path: str, min_bars: int, min_tracks: int) -> list[bool]:
    """Metadata-only filter — no MIDI parsing (avoids C++ crashes on corrupt rows).

    GigaMIDI parquet columns used:
      num_tracks          — track count
      total_notes         — excludes silent/empty files
      loop_duration_beats — conservative bar estimate (floor: 2 beats/bar)
    """
    import pyarrow.compute as pc

    meta = pq.read_table(
        parquet_path,
        columns=["num_tracks", "total_notes", "loop_duration_beats"],
    )

    m_tracks = pc.greater_equal(meta["num_tracks"], min_tracks).to_pylist()
    m_notes  = pc.greater(meta["total_notes"], 0).to_pylist()

    beats_threshold = float(min_bars * 2)
    def _beats_ok(x) -> bool:
        if isinstance(x, list):
            mx = max((float(v) for v in x if v is not None), default=0.0)
        else:
            mx = float(x) if x is not None else 0.0
        return mx >= beats_threshold

    raw_beats = meta["loop_duration_beats"].to_pylist()
    return [t and n and _beats_ok(b) for t, n, b in zip(m_tracks, m_notes, raw_beats)]


def _probe_chunk(
    parquet_path: str,
    chunk: list[int],
    valid_ts: frozenset[str] | None,
) -> list[int]:
    """Parse and validate a chunk in a subprocess; bisect on segfault.

    Checks:
      - Score.from_bytes() succeeds (Python exceptions are tolerated — only
        SIGSEGV kills the subprocess and triggers bisection).
      - If valid_ts is provided, every (ts_num/ts_den) in the parsed score is
        in the allowed set.
    """
    import subprocess, json, os, sys

    valid_ts_list = sorted(valid_ts) if valid_ts is not None else None
    sys_path_repr  = repr(sys.path)
    parquet_repr   = repr(parquet_path)
    chunk_repr     = repr(chunk)
    ts_repr        = repr(valid_ts_list)

    script = "\n".join([
        "import sys, json",
        f"sys.path[:0] = {sys_path_repr}",
        "import datasets as hf",
        "from midigpt._types import Score",
        f"_vts = {ts_repr}",
        "valid_ts = set(_vts) if _vts is not None else None",
        f"data = hf.load_dataset('parquet', data_files={parquet_repr}, split='train')",
        f"chunk = {chunk_repr}",
        "ok = []",
        "for i in chunk:",
        "    try:",
        "        score = Score.from_bytes(data[i]['music'])",
        "        if valid_ts is not None:",
        "            ts_in = {str(b.ts_numerator)+'/'+str(b.ts_denominator) for t in score.tracks for b in t.bars}",
        "            if not ts_in.issubset(valid_ts):",
        "                continue",
        "        ok.append(i)",
        "    except Exception:",
        "        ok.append(i)",
        "print(json.dumps(ok))",
    ])

    env = os.environ.copy()
    env["PYTHONPATH"] = ":".join(sys.path)
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=120, env=env,
    )

    if result.returncode == 0 and result.stdout.strip():
        return json.loads(result.stdout.strip())
    if len(chunk) == 1:
        return []   # single index caused segfault — drop it
    mid = len(chunk) // 2
    return (
        _probe_chunk(parquet_path, chunk[:mid], valid_ts)
        + _probe_chunk(parquet_path, chunk[mid:], valid_ts)
    )


def _load_or_build_valid_indices(
    parquet_path: str,
    min_bars: int,
    min_tracks: int,
    valid_time_sigs: frozenset[str] | None = None,
) -> list[int]:
    """Two-phase filter: fast metadata check then subprocess parse+TS validation.

    Cache key includes encoder time-signature hash so it invalidates when the
    encoder changes.
    """
    import numpy as np

    ts_key = (
        hashlib.md5("|".join(sorted(valid_time_sigs)).encode()).hexdigest()[:8]
        if valid_time_sigs else "nots"
    )
    cache_file = cached_indices(
        file_cache_key(parquet_path, min_bars=min_bars, min_tracks=min_tracks, ts=ts_key)
    )

    if cache_file.exists():
        valid = np.load(cache_file).tolist()
        print(f"  dataset filter: {len(valid)} valid rows loaded from cache")
        return valid

    # Phase 1: fast metadata filter (pure pyarrow, no MIDI parsing)
    print("  dataset filter: phase 1 — metadata check…", flush=True)
    mask = _metadata_valid_mask(parquet_path, min_bars=min_bars, min_tracks=min_tracks)
    candidates = [i for i, v in enumerate(mask) if v]
    print(f"  dataset filter: {len(candidates)}/{len(mask)} pass metadata filter")

    # Phase 2: parse + time-sig validation (subprocess-isolated to catch segfaults)
    print(f"  dataset filter: phase 2 — parse+TS validation ({len(candidates)} rows)…", flush=True)
    chunk_size = 500
    valid: list[int] = []
    try:
        from tqdm.auto import tqdm as _tqdm
        it = _tqdm(range(0, len(candidates), chunk_size), desc="  parse+TS scan")
    except ImportError:
        it = range(0, len(candidates), chunk_size)
    for start in it:
        chunk = candidates[start : start + chunk_size]
        valid.extend(_probe_chunk(parquet_path, chunk, valid_time_sigs))

    dropped = len(candidates) - len(valid)
    print(
        f"  dataset filter: dropped {dropped} rows (parse errors / bad TS) — "
        f"{len(valid)} kept"
    )
    np.save(cache_file, np.array(valid, dtype=np.int64))
    print("  dataset filter: saved to cache")
    return valid


# ── schema for DatasetBuilder (preprocessed format) ───────────────────────────

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


# ── static augments applied once per score before windowing ───────────────────

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

        if infill_probability > 0 and not cfg_dict.get("supports_infill", False):
            raise ValueError(
                f"infill_probability={infill_probability} > 0 but the encoder config "
                f"has supports_infill=false. Set infill_probability=0.0 or use an "
                f"infill-capable checkpoint."
            )
        if mask_bar_config is not None:
            if not tokenizer._vocab.has(_core.TokenType.MaskBar):
                raise ValueError(
                    "mask_bar_config is set but the encoder vocab does not include "
                    "the MaskBar token. Set mask_bar_config=None or use a "
                    "masking-capable checkpoint."
                )

        ts_list = cfg_dict.get("time_signatures")
        valid_ts: frozenset[str] | None = frozenset(ts_list) if ts_list else None

        valid = _load_or_build_valid_indices(
            parquet_path,
            min_bars=self._num_bars_choices[-1],
            min_tracks=min_tracks,
            valid_time_sigs=valid_ts,
        )
        self._data = self._data.select(valid)

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int, _depth: int = 0) -> dict:
        if _depth >= 10:
            raise RuntimeError(f"MidiGPTDataset: failed to load a valid sample after 10 attempts")
        try:
            return self._encode_one(idx)
        except Exception:
            return self.__getitem__(random.randrange(len(self._data)), _depth + 1)

    def _encode_one(self, idx: int) -> dict:
        score = Score.from_bytes(self._data[idx]["music"])
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
                try:
                    tokens = self._tokenizer.encode(window, opts=encode_opts)
                except Exception:
                    continue

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
        try:
            tokens = self._tokenizer.encode(window if window else score)[: self._max_seq_len]
        except Exception:
            raise ValueError(f"Failed to encode sample {idx} after all retries")
        return {"input_ids": tokens, "labels": tokens}
