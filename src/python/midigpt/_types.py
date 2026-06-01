from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class Note:
    pitch:          int
    velocity:       int
    onset_ticks:    int
    duration_ticks: int
    delta:          int = 0

@dataclass
class Bar:
    notes:          list[Note] = field(default_factory=list)
    ts_numerator:   int  = 4
    ts_denominator: int  = 4
    beat_length:    float = 4.0
    future:         bool = False

@dataclass
class Track:
    bars:       list[Bar] = field(default_factory=list)
    instrument: int       = 0
    track_type: str       = "melodic"   # "melodic" | "drum"
    attributes: dict[str, int] = field(default_factory=dict)

@dataclass
class Score:
    tracks:     list[Track] = field(default_factory=list)
    resolution: int         = 480
    tempo:      int         = 500000

    @classmethod
    def from_midi(cls, path: str) -> "Score":
        from midigpt._converters import from_cpp
        import midigpt._core as _core
        return from_cpp(_core.MidiReader().read(path))

    @classmethod
    def from_bytes(cls, data: bytes) -> "Score":
        from midigpt._converters import from_cpp
        import midigpt._core as _core
        return from_cpp(_core.MidiReader().read_bytes(list(data)))

    def to_midi(self, path: str) -> None:
        from midigpt._converters import to_cpp
        import midigpt._core as _core
        _core.MidiWriter().write(to_cpp(self), path)

    @classmethod
    def from_dict(cls, d: dict) -> "Score":
        # Basic serialization logic for datasets
        tracks = []
        for td in d.get("tracks", []):
            bars = []
            for bd in td.get("bars", []):
                notes = [Note(**nd) for nd in bd.get("notes", [])]
                bars.append(Bar(notes=notes, ts_numerator=bd.get("ts_numerator", 4), 
                                ts_denominator=bd.get("ts_denominator", 4), future=bd.get("future", False)))
            tracks.append(Track(bars=bars, instrument=td.get("instrument", 0), track_type=td.get("track_type", "melodic")))
        return cls(tracks=tracks, resolution=d.get("resolution", 480), tempo=d.get("tempo", 500000))

    def to_dict(self) -> dict:
        return {
            "resolution": self.resolution,
            "tempo": self.tempo,
            "tracks": [{
                "instrument": t.instrument,
                "track_type": t.track_type,
                "bars": [{
                    "ts_numerator": b.ts_numerator,
                    "ts_denominator": b.ts_denominator,
                    "future": b.future,
                    "notes": [{"pitch": n.pitch, "velocity": n.velocity, "onset_ticks": n.onset_ticks, 
                               "duration_ticks": n.duration_ticks, "delta": n.delta} for n in b.notes]
                } for b in t.bars]
            } for t in self.tracks]
        }
