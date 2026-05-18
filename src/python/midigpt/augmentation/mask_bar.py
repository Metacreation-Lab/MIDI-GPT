import random
from dataclasses import dataclass
from midigpt._types import Score
from midigpt.augmentation.base import BaseTransform


@dataclass
class MaskBarConfig:
    apply_probability: float = 0.5
    mode: int = 2
    bar_fraction: float = 0.25
    max_lookahead: int = 4


class MaskBar(BaseTransform):
    """Stochastic bar masking for training data.

    Marks selected bars as `future=True` so the encoder emits MASK_BAR tokens.
    Three modes:
      0 — Random: shuffle eligible bars, mask a random fraction
      1 — Structured-future: pick a time position and mask lookahead bars
          on conditioning tracks (mirrors inference-time pattern)
      2 — Mixed: 50/50 choice between mode 0 and 1
    """

    def __init__(self, config: MaskBarConfig | None = None,
                 infill_bars: set[tuple[int, int]] | None = None):
        self.cfg = config or MaskBarConfig()
        self._infill = infill_bars or set()

    def __call__(self, score: Score) -> Score:
        if random.random() >= self.cfg.apply_probability:
            return score

        num_tracks = len(score.tracks)
        num_bars = max((len(t.bars) for t in score.tracks), default=0)
        if num_bars == 0:
            return score

        mode = self.cfg.mode
        if mode == 2:
            mode = random.choice([0, 1])

        if mode == 0:
            self._random_mask(score, num_tracks, num_bars)
        else:
            self._structured_future_mask(score, num_tracks, num_bars)
        return score

    def _random_mask(self, score: Score, num_tracks: int, num_bars: int) -> None:
        eligible = []
        for t in range(num_tracks):
            for b in range(len(score.tracks[t].bars)):
                if (t, b) not in self._infill:
                    eligible.append((t, b))
        if not eligible:
            return
        random.shuffle(eligible)
        max_count = max(1, int(self.cfg.bar_fraction * len(eligible)))
        count = random.randint(1, max_count)
        for t, b in eligible[:count]:
            score.tracks[t].bars[b].future = True

    def _structured_future_mask(self, score: Score, num_tracks: int,
                                num_bars: int) -> None:
        if num_bars < 2:
            return
        t_cur = random.randint(0, num_bars - 2)
        k = random.randint(1, max(1, self.cfg.max_lookahead))
        end = min(t_cur + k, num_bars - 1)

        for t_idx in range(num_tracks):
            track = score.tracks[t_idx]
            is_gen_track = any((t_idx, b) in self._infill
                               for b in range(len(track.bars)))
            if is_gen_track:
                continue
            if random.random() < 0.25:
                continue
            for b in range(t_cur + 1, end + 1):
                if b < len(track.bars) and (t_idx, b) not in self._infill:
                    track.bars[b].future = True
