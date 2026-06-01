from midigpt._types import Score
from midigpt.attributes.base import BaseAttribute


class KeySignature(BaseAttribute):
    name = "key_signature"
    token_type = "KeySignature"
    level = "track"
    track_type = "melodic"
    size = 25

    def compute(self, score: Score, track_idx: int, bar_idx: int | None = None) -> float | int:
        pitch_class_counts = [0.0] * 12
        note_weight = 0
        for track in score.tracks:
            for bar in track.bars:
                for note in bar.notes:
                    if note.velocity > 0:
                        pitch_class_counts[note.pitch % 12] += note.duration_ticks
                        note_weight += note.duration_ticks

        if note_weight == 0:
            return 24  # no key

        weights = [
            6.35,
            2.23,
            3.48,
            2.33,
            4.38,
            4.09,
            2.52,
            5.19,
            2.39,
            3.66,
            2.29,
            2.88,
            6.33,
            2.68,
            3.52,
            5.38,
            2.60,
            3.53,
            2.54,
            4.75,
            3.98,
            2.69,
            3.34,
            3.17,
        ]

        solution = [0.0] * 24
        for i in range(12):
            for j in range(12):
                solution[i] += weights[(j - i + 12) % 12] * pitch_class_counts[j]
                solution[i + 12] += weights[(j - i + 12) % 12 + 12] * pitch_class_counts[j]

        max_index = solution.index(max(solution))
        return max_index

    def quantize(self, value: float | int) -> int:
        return max(0, min(24, int(value)))
