#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "../../src/cpp/tokenizer/encoder.h"
#include "../../src/cpp/tokenizer/decoder.h"

using namespace midigpt;
using namespace midigpt::tokenizer;

TEST_CASE("Encoder & Decoder roundtrip") {
    EncoderConfig config;
    config.resolution = 480;
    config.token_domains.push_back({TokenType::PieceStart, 1});
    config.token_domains.push_back({TokenType::PieceEnd, 1});
    config.token_domains.push_back({TokenType::Track, 128});
    config.token_domains.push_back({TokenType::Instrument, 128});
    config.token_domains.push_back({TokenType::TrackEnd, 1});
    config.token_domains.push_back({TokenType::Bar, 1});
    config.token_domains.push_back({TokenType::BarEnd, 1});
    config.token_domains.push_back({TokenType::TimeSig, 36});
    config.token_domains.push_back({TokenType::TimeAbsolutePos, 192});
    config.token_domains.push_back({TokenType::VelocityLevel, 32});
    config.token_domains.push_back({TokenType::NoteOnset, 128}); // Pitch
    config.token_domains.push_back({TokenType::NoteDuration, 128});
    
    Vocabulary vocab(config);
    Encoder encoder(vocab);
    Decoder decoder(vocab);
    
    Score original;
    original.resolution = 480;
    
    Note n1;
    n1.onset_ticks = 48; // TimeAbsolutePos -> 48
    n1.duration_ticks = 48; // NoteDuration -> 48
    n1.pitch = 60; // NoteOnset -> 60
    n1.velocity = 64; // VelocityLevel -> mapped to 16
    original.notes.push_back(n1);
    
    Bar b1;
    b1.note_indices.push_back(0);
    b1.ts_numerator = 4;
    b1.ts_denominator = 4;
    
    Track t1;
    t1.instrument = 0; // mapped to 0
    t1.bars.push_back(b1);
    original.tracks.push_back(t1);
    
    auto tokens = encoder.encode(original);
    
    // We expect:
    // PieceStart
    // Track
    // Instrument
    // Bar
    // TimeSig
    // TimeAbsolutePos
    // VelocityLevel
    // NoteOnset (Pitch)
    // NoteDuration
    // BarEnd
    // TrackEnd
    // PieceEnd
    CHECK(tokens.size() == 12);
    
    Score decoded = decoder.decode(tokens);
    
    CHECK(decoded.resolution == 480);
    CHECK(decoded.tracks.size() == 1);
    CHECK(decoded.tracks[0].bars.size() == 1);
    CHECK(decoded.tracks[0].bars[0].ts_numerator == 4);
    CHECK(decoded.notes.size() == 1);
    CHECK(decoded.notes[0].onset_ticks == 48);
    CHECK(decoded.notes[0].duration_ticks == 48);
    CHECK(decoded.notes[0].pitch == 60);
    CHECK(decoded.notes[0].velocity == 66); // 16 mapped back to 66
}
