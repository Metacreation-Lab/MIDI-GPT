import random
from midigpt._types import Score
from midigpt.augmentation.base import BaseTransform

class BarWindow(BaseTransform):
    def __init__(self, num_bars: int):
        self.num_bars = num_bars

    def __call__(self, score: Score) -> Score:
        if not score.tracks:
            return score
            
        max_bars = max((len(t.bars) for t in score.tracks), default=0)
        if max_bars <= self.num_bars:
            return score
            
        start_idx = random.randint(0, max_bars - self.num_bars)
        
        for track in score.tracks:
            end_idx = min(len(track.bars), start_idx + self.num_bars)
            track.bars = track.bars[start_idx:end_idx]
            
        return score
