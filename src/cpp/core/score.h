#pragma once

#include <vector>
#include <map>
#include <string>
#include "types.h"

namespace midigpt {

struct Note {
    int pitch;
    int velocity;
    int onset_ticks;
    int duration_ticks;
    int delta = 0;          // microtiming offset from onset
};

struct Bar {
    std::vector<int> note_indices;  // indices into Score::notes pool
    int  ts_numerator   = 4;
    int  ts_denominator = 4;
    double beat_length  = 0;        // beats per bar
    bool has_notes      = false;
    bool future         = false;    // true → encode as MASK_BAR
};

struct Track {
    std::vector<Bar> bars;
    int       instrument = 0;
    TrackType type       = TrackType::Melodic;
    std::map<std::string, int> attributes; // e.g. "note_density" -> 5
};

struct Score {
    std::vector<Track> tracks;
    std::vector<Note>  notes;   // global pool; Bars index into this
    int resolution = 480;       // ticks per quarter note
    int tempo      = 500000;    // microseconds per beat
};

} // namespace midigpt
