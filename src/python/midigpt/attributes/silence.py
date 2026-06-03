from midigpt._types import Score
from midigpt.attributes.base import BaseAttribute


class SilenceProportion(BaseAttribute):
    size = 10

    def __init__(self, level: str = "track", track_type: str = "both"):
        self.level = level
        self.track_type = track_type
        self.name = "bar_silence_proportion" if level == "bar" else "silence_proportion"
        self.token_type = "SilenceProportionBar" if level == "bar" else "SilenceProportion"

    def _bar_silence(self, bar, bar_start_tick: int, bar_len: int) -> float:
        if bar_len == 0:
            return 0.0
        if not bar.notes:
            return 1.0
        active_ticks = [False] * bar_len
        for note in bar.notes:
            rel_start = note.onset_ticks - bar_start_tick
            start = max(0, min(bar_len, rel_start))
            end = max(0, min(bar_len, rel_start + note.duration_ticks))
            for i in range(start, end):
                active_ticks[i] = True
        return active_ticks.count(False) / bar_len

    def compute(self, score: Score, track_idx: int, bar_idx: int | None = None) -> float | int:
        track = score.tracks[track_idx]
        if not track.bars:
            return 0.0

        bar_start_tick = 0
        if self.level == "bar" and bar_idx is not None:
            for i in range(bar_idx):
                bar_len_i = int(
                    track.bars[i].ts_numerator * (score.resolution * 4 / track.bars[i].ts_denominator)
                )
                bar_start_tick += bar_len_i
            if bar_idx >= len(track.bars):
                return 0.0
            bar = track.bars[bar_idx]
            bar_len = int(bar.ts_numerator * (score.resolution * 4 / bar.ts_denominator))
            return self._bar_silence(bar, bar_start_tick, bar_len)

        # track-level: average silence proportion across all bars
        silence_props = []
        for bar in track.bars:
            bar_len = int(bar.ts_numerator * (score.resolution * 4 / bar.ts_denominator))
            if bar_len == 0:
                bar_start_tick += bar_len
                continue
            silence_props.append(self._bar_silence(bar, bar_start_tick, bar_len))
            bar_start_tick += bar_len

        return sum(silence_props) / len(silence_props) if silence_props else 0.0

    def quantize(self, value: float | int) -> int:
        return max(0, min(9, int(value * 10)))
