from midigpt._types import Score
from midigpt.attributes.base import BaseAttribute


class SilenceProportion(BaseAttribute):
    name = "silence_proportion"
    token_type = "SilenceProportion"
    level = "track"
    track_type = "both"
    size = 10

    def compute(self, score: Score, track_idx: int, bar_idx: int | None = None) -> float | int:
        track = score.tracks[track_idx]
        if not track.bars:
            return 0.0

        silence_props = []
        bar_start_tick = 0

        for bar in track.bars:
            # bar duration in ticks
            bar_len = int(bar.ts_numerator * (score.resolution * 4 / bar.ts_denominator))
            if bar_len == 0:
                continue

            if not bar.notes:
                silence_props.append(1.0)
                bar_start_tick += bar_len
                continue

            active_ticks = [False] * bar_len
            for note in bar.notes:
                # onset_ticks is absolute, so make it relative to bar
                rel_start = note.onset_ticks - bar_start_tick
                start = max(0, min(bar_len, rel_start))
                end = max(0, min(bar_len, rel_start + note.duration_ticks))
                for i in range(start, end):
                    active_ticks[i] = True

            silent_count = active_ticks.count(False)
            silence_props.append(silent_count / bar_len)
            bar_start_tick += bar_len

        if not silence_props:
            return 0.0

        # Return average silence proportion (0.0 to 1.0)
        return sum(silence_props) / len(silence_props)

    def quantize(self, value: float | int) -> int:
        # Map [0.0, 1.0] -> [0, 9] (domain_size is 10)
        val = int(value * 10)
        # If exactly 1.0, it becomes 10, so we clamp to 9
        return max(0, min(9, val))
