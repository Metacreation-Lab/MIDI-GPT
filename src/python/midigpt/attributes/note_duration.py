import math
from typing import Optional
from midigpt._types import Score
from midigpt.attributes.base import BaseAttribute

class NoteDurationDistribution(BaseAttribute):
    name       = "note_duration_dist"
    token_type = "NoteDurationDist"
    level      = "track"
    track_type = "both"
    size       = 6

    def compute(self, score: Score, track_idx: int, bar_idx: Optional[int] = None) -> float | int:
        track = score.tracks[track_idx]
        durations = []
        for bar in track.bars:
            for note in bar.notes:
                d = note.duration_ticks
                if d <= 0:
                    continue
                # log2 duration level: 32nd=0, 16th=1, 8th=2, quarter=3, half=4, whole=5
                # Using the old logic: log2(max(d / 3.0, 1e-6))
                # assuming 24 ticks per 32nd note ? Wait, if resolution=480:
                # whole = 1920
                # half = 960
                # quarter = 480
                # 8th = 240
                # 16th = 120
                # 32nd = 60
                # For quarter (480), d/60 = 8. log2(8) = 3. Yes, this works!
                # We divide by (score.resolution / 8) which is 480/8 = 60
                val = d / (score.resolution / 8.0)
                level = max(0, min(5, int(math.log2(max(val, 1.0)))))
                durations.append(level)
                
        if not durations:
            return 3 # default to quarter notes
            
        # find the most common duration bin
        from collections import Counter
        return Counter(durations).most_common(1)[0][0]

    def quantize(self, value: float | int) -> int:
        return max(0, min(5, int(value)))
