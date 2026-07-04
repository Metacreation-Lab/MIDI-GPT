import statistics

from midigpt._types import Score
from midigpt.attributes.base import BaseAttribute

# Direct port of GigaMIDI's loops_nomml/nomml.py (Metacreation-Lab/GigaMIDI-Dataset).
# Classifies how far a note's onset sits from common rhythmic subdivisions of
# one quarter note. Level 12 (no match down to 1/32-note / triplet-1/16
# granularity) marks "expressive" (freely-timed) performance; levels 0-11
# mark increasingly coarse quantization to a rational grid.


def get_metric_depth(time: int, tpq: float, max_depth: int = 6) -> int:
    for i in range(max_depth):
        period = tpq / (2**i)
        if period > 0 and time % period == 0:
            return 2 * i
    for i in range(max_depth):
        period = tpq * 2 / ((2**i) * 3)
        if period > 0 and time % period == 0:
            return 2 * i + 1
    return max_depth * 2


def compute_median_metric_depth(score: Score, track_idx: int) -> int:
    """Median metric depth for one track, using score.resolution as tpq.

    For results comparable to GigaMIDI's published NOMML values, call this on
    a Score at (or close to) its original read resolution -- not one that has
    already been downsampled to a coarse encoder grid, since quantization
    destroys the fine onset alignment this metric depends on.
    """
    track = score.tracks[track_idx]
    depths = [
        get_metric_depth(note.onset_ticks, score.resolution)
        for bar in track.bars
        for note in bar.notes
    ]
    if not depths:
        return 12
    return int(statistics.median(depths))


def nomml_per_track(score: Score) -> list[int]:
    """Median metric depth for every track in a freshly-loaded Score.

    Intended for inference-time conditioning on arbitrary MIDI (no GigaMIDI
    metadata available): call this on the Score returned by
    Score.from_midi()/from_bytes() *before* Tokenizer.encode() downsamples it,
    then stash results into track.attributes["_nomml_raw"] per track so
    Nomml.compute() picks them up instead of recomputing on quantized data.
    """
    return [compute_median_metric_depth(score, i) for i in range(len(score.tracks))]


class Nomml(BaseAttribute):
    name = "nomml"
    token_type = "TrackLevelNomml"
    level = "track"
    track_type = "both"
    size = 13  # 0-12 ordinal median metric depth

    def compute(self, score: Score, track_idx: int, bar_idx: int | None = None) -> float | int:
        precomputed = score.tracks[track_idx].attributes.get("_nomml_raw")
        if precomputed is not None:
            return precomputed
        return compute_median_metric_depth(score, track_idx)

    def quantize(self, value: float | int) -> int:
        return max(0, min(12, int(round(value))))

    def dequantize(self, quantized: int) -> float | int:
        return quantized
