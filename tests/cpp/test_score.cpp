#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "../../src/cpp/core/types.h"
#include "../../src/cpp/core/score.h"

using namespace midigpt;

TEST_CASE("Score construction and defaults") {
    Score score;
    CHECK(score.resolution == 480);
    CHECK(score.tempo == 500000);
    CHECK(score.tracks.empty());
    CHECK(score.notes.empty());
}

TEST_CASE("Note pool indexing and copy semantics") {
    Score score;
    
    // Add some notes
    score.notes.push_back({60, 100, 0, 480, 0});
    score.notes.push_back({62, 100, 480, 480, 0});
    
    // Create a track and bar
    Track track;
    track.type = TrackType::Melodic;
    
    Bar bar;
    bar.note_indices.push_back(0);
    bar.note_indices.push_back(1);
    bar.has_notes = true;
    
    track.bars.push_back(bar);
    score.tracks.push_back(track);
    
    CHECK(score.tracks.size() == 1);
    CHECK(score.tracks[0].bars.size() == 1);
    CHECK(score.tracks[0].bars[0].note_indices.size() == 2);
    
    // Access notes via pool
    int first_note_idx = score.tracks[0].bars[0].note_indices[0];
    int second_note_idx = score.tracks[0].bars[0].note_indices[1];
    
    CHECK(score.notes[first_note_idx].pitch == 60);
    CHECK(score.notes[second_note_idx].pitch == 62);
    
    // Copy semantics
    Score score_copy = score;
    CHECK(score_copy.notes.size() == 2);
    
    // Modifying the copy doesn't modify the original
    score_copy.notes[0].pitch = 61;
    CHECK(score.notes[0].pitch == 60);
    CHECK(score_copy.notes[0].pitch == 61);
}
