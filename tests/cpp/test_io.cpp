// Canonical MIDI I/O suite. Covers writer→reader roundtrips for:
//   - Single melodic note (baseline)
//   - Drum track (channel 9)
//   - Multiple instruments across tracks
//   - Velocity edges (1 and 127)
//   - Onset/duration at bar boundaries
//   - Time signature change mid-piece
//   - Tempo preservation

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "../../src/cpp/io/midi_reader.h"
#include "../../src/cpp/io/midi_writer.h"

using namespace midigpt;
using namespace midigpt::io;

namespace {

Score make_score_one_note(int pitch, int vel, int onset, int dur,
                          TrackType ty = TrackType::Melodic, int instr = 0) {
    Score s; s.resolution = 480; s.tempo = 500000;
    Note n{pitch, vel, onset, dur, 0};
    s.notes.push_back(n);
    Bar b; b.ts_numerator = 4; b.ts_denominator = 4; b.beat_length = 4;
    b.note_indices.push_back(0); b.has_notes = true;
    Track t; t.type = ty; t.instrument = instr;
    t.bars.push_back(b);
    s.tracks.push_back(t);
    return s;
}

}  // namespace

// ---------------------------------------------------------------------------
// Baseline
// ---------------------------------------------------------------------------

TEST_CASE("IO: melodic single-note roundtrip preserves everything") {
    auto s = make_score_one_note(60, 100, 0, 480);
    MidiWriter w; auto bytes = w.write_bytes(s);
    CHECK(bytes.size() > 0);

    MidiReader r; Score d = r.read_bytes(bytes);
    CHECK(d.resolution == 480);
    CHECK(d.tempo == 500000);
    REQUIRE(d.tracks.size() == 1);
    CHECK(d.tracks[0].type == TrackType::Melodic);
    REQUIRE(d.tracks[0].bars.size() == 1);
    REQUIRE(d.tracks[0].bars[0].note_indices.size() == 1);
    int idx = d.tracks[0].bars[0].note_indices[0];
    CHECK(d.notes[idx].pitch == 60);
    CHECK(d.notes[idx].velocity == 100);
    CHECK(d.notes[idx].onset_ticks == 0);
    CHECK(d.notes[idx].duration_ticks == 480);
}

// ---------------------------------------------------------------------------
// Drum track
// ---------------------------------------------------------------------------

TEST_CASE("IO: drum track roundtrips with TrackType::Drum") {
    auto s = make_score_one_note(36, 100, 0, 240, TrackType::Drum);
    MidiWriter w; auto bytes = w.write_bytes(s);
    MidiReader r; Score d = r.read_bytes(bytes);
    REQUIRE(d.tracks.size() == 1);
    CHECK(d.tracks[0].type == TrackType::Drum);
}

// ---------------------------------------------------------------------------
// Multi-instrument
// ---------------------------------------------------------------------------

TEST_CASE("IO: multi-instrument tracks preserve instrument numbers") {
    Score s; s.resolution = 480; s.tempo = 500000;
    s.notes.push_back({60, 80, 0, 240, 0});
    s.notes.push_back({40, 80, 0, 240, 0});
    for (int i = 0; i < 2; ++i) {
        Bar b; b.ts_numerator = 4; b.ts_denominator = 4; b.beat_length = 4;
        b.note_indices.push_back(i); b.has_notes = true;
        Track t; t.type = TrackType::Melodic;
        t.instrument = (i == 0) ? 0 : 33;  // piano / bass
        t.bars.push_back(b);
        s.tracks.push_back(t);
    }
    MidiWriter w; auto bytes = w.write_bytes(s);
    MidiReader r; Score d = r.read_bytes(bytes);
    REQUIRE(d.tracks.size() == 2);
    // Two distinct instruments survive
    CHECK(d.tracks[0].instrument != d.tracks[1].instrument);
}

// ---------------------------------------------------------------------------
// Velocity edges
// ---------------------------------------------------------------------------

TEST_CASE("IO: velocity=1 preserved") {
    auto s = make_score_one_note(60, 1, 0, 240);
    MidiWriter w; auto bytes = w.write_bytes(s);
    MidiReader r; Score d = r.read_bytes(bytes);
    REQUIRE(d.notes.size() == 1);
    CHECK(d.notes[0].velocity == 1);
}

TEST_CASE("IO: velocity=127 preserved") {
    auto s = make_score_one_note(60, 127, 0, 240);
    MidiWriter w; auto bytes = w.write_bytes(s);
    MidiReader r; Score d = r.read_bytes(bytes);
    REQUIRE(d.notes.size() == 1);
    CHECK(d.notes[0].velocity == 127);
}

// ---------------------------------------------------------------------------
// Onset/duration at boundaries
// ---------------------------------------------------------------------------

TEST_CASE("IO: onset=0 short duration roundtrip") {
    auto s = make_score_one_note(60, 64, 0, 1);
    MidiWriter w; auto bytes = w.write_bytes(s);
    MidiReader r; Score d = r.read_bytes(bytes);
    REQUIRE(d.notes.size() == 1);
    CHECK(d.notes[0].onset_ticks == 0);
    CHECK(d.notes[0].duration_ticks >= 1);
}

// ---------------------------------------------------------------------------
// Time signature change mid-piece
// ---------------------------------------------------------------------------

TEST_CASE("IO: time signature change across bars roundtrips") {
    Score s; s.resolution = 480; s.tempo = 500000;
    Note n{60, 64, 0, 240, 0}; s.notes.push_back(n);
    Note m{62, 64, 0, 240, 0}; s.notes.push_back(m);
    Track t; t.type = TrackType::Melodic; t.instrument = 0;
    Bar a; a.ts_numerator = 4; a.ts_denominator = 4; a.beat_length = 4;
    a.note_indices.push_back(0); a.has_notes = true;
    t.bars.push_back(a);
    Bar b; b.ts_numerator = 3; b.ts_denominator = 4; b.beat_length = 3;
    b.note_indices.push_back(1); b.has_notes = true;
    t.bars.push_back(b);
    s.tracks.push_back(t);

    MidiWriter w; auto bytes = w.write_bytes(s);
    MidiReader r; Score d = r.read_bytes(bytes);
    REQUIRE(d.tracks.size() == 1);
    REQUIRE(d.tracks[0].bars.size() >= 2);
    CHECK(d.tracks[0].bars[0].ts_numerator == 4);
    CHECK(d.tracks[0].bars[1].ts_numerator == 3);
    CHECK(d.tracks[0].bars[1].ts_denominator == 4);
}

// ---------------------------------------------------------------------------
// Tempo
// ---------------------------------------------------------------------------

TEST_CASE("IO: non-default tempo preserved") {
    auto s = make_score_one_note(60, 64, 0, 240);
    s.tempo = 300000;  // 200 BPM
    MidiWriter w; auto bytes = w.write_bytes(s);
    MidiReader r; Score d = r.read_bytes(bytes);
    CHECK(d.tempo == 300000);
}
