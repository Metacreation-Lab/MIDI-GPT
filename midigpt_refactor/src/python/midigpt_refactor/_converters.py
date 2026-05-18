import midigpt_refactor._core as _core
from midigpt_refactor._types import Score, Track, Bar, Note

def to_cpp(score: Score | _core.Score) -> _core.Score:
    if isinstance(score, _core.Score):
        return score
        
    cpp = _core.Score()
    cpp.resolution = score.resolution
    cpp.tempo      = score.tempo
    note_pool: list[Note] = []
    
    cpp_tracks = []
    for track in score.tracks:
        cpp_track = _core.Track()
        cpp_track.instrument = track.instrument
        cpp_track.type = (_core.TrackType.Drum
                          if getattr(track, "track_type", None) == "drum"
                          else _core.TrackType.Melodic)
        cpp_bars = []
        for bar in track.bars:
            cpp_bar = _core.Bar()
            cpp_bar.ts_numerator   = bar.ts_numerator
            cpp_bar.ts_denominator = bar.ts_denominator
            cpp_bar.beat_length    = bar.beat_length
            cpp_bar.future         = bar.future
            cpp_note_indices = []
            for note in bar.notes:
                cpp_note_indices.append(len(note_pool))
                note_pool.append(note)
            cpp_bar.note_indices = cpp_note_indices
            cpp_bars.append(cpp_bar)
        cpp_track.bars = cpp_bars
        cpp_track.attributes = track.attributes
        cpp_tracks.append(cpp_track)
    cpp.tracks = cpp_tracks
        
    cpp_notes = []
    for note in note_pool:
        cpp_note = _core.Note()
        cpp_note.pitch          = note.pitch
        cpp_note.velocity       = note.velocity
        cpp_note.onset_ticks    = note.onset_ticks
        cpp_note.duration_ticks = note.duration_ticks
        cpp_note.delta          = note.delta
        cpp_notes.append(cpp_note)
    cpp.notes = cpp_notes
        
    return cpp

def from_cpp(cpp: _core.Score) -> Score:
    pool = [Note(n.pitch, n.velocity, n.onset_ticks, n.duration_ticks, n.delta)
            for n in cpp.notes]
    tracks = []
    for ct in cpp.tracks:
        bars = []
        for cb in ct.bars:
            bars.append(Bar(
                notes          = [pool[i] for i in cb.note_indices],
                ts_numerator   = cb.ts_numerator,
                ts_denominator = cb.ts_denominator,
                beat_length    = cb.beat_length,
                future         = cb.future,
            ))
        tracks.append(Track(
            bars       = bars,
            instrument = ct.instrument,
            track_type = "drum" if ct.type == _core.TrackType.Drum else "melodic",
            attributes = dict(ct.attributes),
        ))
    return Score(tracks=tracks, resolution=cpp.resolution, tempo=cpp.tempo)
