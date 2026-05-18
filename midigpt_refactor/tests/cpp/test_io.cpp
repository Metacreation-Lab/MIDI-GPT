#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "../../src/cpp/io/midi_reader.h"
#include "../../src/cpp/io/midi_writer.h"

using namespace midigpt;
using namespace midigpt::io;

TEST_CASE("IO roundtrip") {
    // Create a basic score
    Score score;
    score.resolution = 480;
    score.tempo = 500000;
    
    Track track;
    track.type = TrackType::Melodic;
    track.instrument = 0;
    
    Bar bar;
    bar.ts_numerator = 4;
    bar.ts_denominator = 4;
    bar.beat_length = 480 * 4;
    
    score.notes.push_back({60, 100, 0, 480, 0});
    bar.note_indices.push_back(0);
    bar.has_notes = true;
    
    track.bars.push_back(bar);
    score.tracks.push_back(track);
    
    MidiWriter writer;
    auto bytes = writer.write_bytes(score);
    
    CHECK(bytes.size() > 0);
    
    MidiReader reader;
    Score decoded = reader.read_bytes(bytes);
    
    CHECK(decoded.resolution == 480);
    CHECK(decoded.tempo == 500000);
    CHECK(decoded.tracks.size() == 1);
    CHECK(decoded.tracks[0].type == TrackType::Melodic);
    CHECK(decoded.tracks[0].bars.size() == 1);
    CHECK(decoded.tracks[0].bars[0].note_indices.size() == 1);
    
    int note_idx = decoded.tracks[0].bars[0].note_indices[0];
    CHECK(decoded.notes[note_idx].pitch == 60);
    CHECK(decoded.notes[note_idx].velocity == 100);
    CHECK(decoded.notes[note_idx].onset_ticks == 0);
    CHECK(decoded.notes[note_idx].duration_ticks == 480);
}
