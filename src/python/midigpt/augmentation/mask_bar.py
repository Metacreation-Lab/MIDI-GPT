import enum
import random
from dataclasses import dataclass

from midigpt._types import Score
from midigpt.augmentation.base import BaseTransform


class MaskMode(enum.IntEnum):
    """Strategy for selecting which bars to mark as MASK_BAR."""

    RANDOM = 0  # shuffle eligible bars, mask a random fraction
    STRUCTURED = 1  # pick a time position, mask a contiguous lookahead window
    MIXED = 2  # 50/50 between RANDOM and STRUCTURED each call


@dataclass
class MaskBarConfig:
    apply_probability: float = 0.5  # gate: skip masking entirely this sample
    mode: MaskMode = MaskMode.MIXED
    bar_fraction: float = 0.25  # max fraction of eligible bars to mask
    max_lookahead: int = 4  # structured mode: max bars ahead to mask


class MaskBar(BaseTransform):
    """Stochastic bar masking.

    Marks selected bars as ``future=True`` so the C++ encoder emits MASK_BAR
    tokens, hiding those bars from the model's context.

    Invariant (enforced here and in the encoder): a bar passed via
    ``infill_bars`` is an infill *target* (FillInStart/FillInEnd) and must
    never also be masked — the two states are mutually exclusive.

    Args:
        config: masking hyper-parameters.
        infill_bars: set of (track_idx, bar_idx) pairs that are infill targets
            for this sample. MaskBar will skip these bars entirely.
    """

    def __init__(
        self,
        config: MaskBarConfig | None = None,
        infill_bars: set[tuple[int, int]] | None = None,
    ):
        self.cfg = config or MaskBarConfig()
        self._infill = infill_bars or set()

    def __call__(self, score: Score) -> Score:
        if random.random() >= self.cfg.apply_probability:
            return score

        num_bars = max((len(t.bars) for t in score.tracks), default=0)
        if num_bars == 0:
            return score

        mode = self.cfg.mode
        if mode == MaskMode.MIXED:
            mode = random.choice([MaskMode.RANDOM, MaskMode.STRUCTURED])

        if mode == MaskMode.RANDOM:
            self._random_mask(score, num_bars)
        else:
            self._structured_mask(score, num_bars)
        return score

    def _random_mask(self, score: Score, num_bars: int) -> None:
        eligible = [
            (t, b)
            for t in range(len(score.tracks))
            for b in range(len(score.tracks[t].bars))
            if (t, b) not in self._infill
        ]
        if not eligible:
            return
        random.shuffle(eligible)
        max_count = max(1, int(self.cfg.bar_fraction * len(eligible)))
        for t, b in eligible[: random.randint(1, max_count)]:
            score.tracks[t].bars[b].future = True

    def _structured_mask(self, score: Score, num_bars: int) -> None:
        if num_bars < 2:
            return
        t_cur = random.randint(0, num_bars - 2)
        k = random.randint(1, max(1, self.cfg.max_lookahead))
        end = min(t_cur + k, num_bars - 1)

        for t_idx, track in enumerate(score.tracks):
            if random.random() < 0.25:
                continue
            for b in range(t_cur + 1, end + 1):
                if b < len(track.bars) and (t_idx, b) not in self._infill:
                    track.bars[b].future = True
