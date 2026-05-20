"""Score windowing: pick a contiguous bar range and a random track subset."""
from __future__ import annotations
import copy
import math
import random
from typing import Optional

from midigpt._types import Score, Track


def select_window(
    score: Score,
    n_bars: int,
    n_tracks: int,
    min_fill_ratio: float = 0.75,
    rng: Optional[random.Random] = None,
) -> Optional[Score]:
    """Return a Score slice of exactly `n_bars` bars on exactly `n_tracks` tracks.

    A track is "valid" for a window starting at `start` if it has at least
    ceil(min_fill_ratio * n_bars) non-empty bars in [start, start + n_bars).

    Returns None when no valid (start, track_subset) combination exists for
    the requested parameters — the caller should fall back to smaller values.
    """
    _rng = rng or random
    min_nonempty = max(1, math.ceil(min_fill_ratio * n_bars))

    # Build list of valid (start, [track_indices]) pairs
    total_bars = min(
        (len(t.bars) for t in score.tracks),
        default=0,
    )
    if total_bars < n_bars:
        return None

    valid_starts: list[tuple[int, list[int]]] = []
    for start in range(total_bars - n_bars + 1):
        good_tracks = []
        for t_idx, track in enumerate(score.tracks):
            if len(track.bars) < start + n_bars:
                continue
            window_bars = track.bars[start : start + n_bars]
            nonempty = sum(1 for b in window_bars if b.notes)
            if nonempty >= min_nonempty:
                good_tracks.append(t_idx)
        if len(good_tracks) >= n_tracks:
            valid_starts.append((start, good_tracks))

    if not valid_starts:
        return None

    start, good_tracks = _rng.choice(valid_starts)
    _rng.shuffle(good_tracks)
    selected_indices = good_tracks[:n_tracks]

    new_tracks = [
        Track(
            bars=copy.deepcopy(score.tracks[t_idx].bars[start : start + n_bars]),
            instrument=score.tracks[t_idx].instrument,
            track_type=score.tracks[t_idx].track_type,
            attributes=dict(score.tracks[t_idx].attributes),
        )
        for t_idx in selected_indices
    ]
    return Score(tracks=new_tracks, resolution=score.resolution, tempo=score.tempo)
