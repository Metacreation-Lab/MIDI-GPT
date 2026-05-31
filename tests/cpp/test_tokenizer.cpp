// Canonical encoder + decoder suite. Covers:
//   - Single-track roundtrip (basic happy path)
//   - Multi-bar / multi-track scores
//   - Drum vs. melodic tracks (NoteDuration emission differs)
//   - velocity_sticky on/off (header-level flag)
//   - emit_delta_tokens true (Delta/DeltaDirection in vocab)
//   - Multi-fill infill (FILL_IN_PLACEHOLDER + FILL_IN blocks)
//   - use_span_masks (hidden_spans metadata)
//   - remove_future_bars
//   - Partial-encode (suffix-AR)
// Raw assertions only — no production code changes.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "../../src/cpp/tokenizer/encoder.h"
#include "../../src/cpp/tokenizer/decoder.h"
#include "../../src/cpp/tokenizer/encoder_config.h"
#include "../../src/cpp/tokenizer/vocabulary.h"
#include "../../src/cpp/core/score.h"

#include <algorithm>

using namespace midigpt;
using namespace midigpt::tokenizer;

namespace {

// Standard structural vocab used by most cases.
EncoderConfig std_config(bool with_velocity = true, bool with_duration = true,
                         bool with_delta = false, bool with_fillin = false) {
    EncoderConfig cfg;
    cfg.resolution = 480;
    cfg.velocity_levels = 32;
    auto push = [&](TokenType t, int s) { cfg.token_domains.push_back({t, s}); };
    push(TokenType::PieceStart, 1);
    push(TokenType::PieceEnd, 1);
    push(TokenType::Track, 128);
    push(TokenType::Instrument, 128);
    push(TokenType::TrackEnd, 1);
    push(TokenType::Bar, 1);
    push(TokenType::BarEnd, 1);
    push(TokenType::TimeSig, 36);
    push(TokenType::TimeAbsolutePos, 192);
    if (with_velocity) push(TokenType::VelocityLevel, 32);
    push(TokenType::NoteOnset, 128);
    if (with_duration) push(TokenType::NoteDuration, 128);
    if (with_delta) {
        push(TokenType::Delta, 64);
        push(TokenType::DeltaDirection, 2);
    }
    if (with_fillin) {
        push(TokenType::FillInPlaceholder, 1);
        push(TokenType::FillInStart, 1);
        push(TokenType::FillInEnd, 1);
    }
    return cfg;
}

Score one_bar_one_note(int pitch = 60, int vel = 64, int onset = 48,
                       int dur = 48) {
    Score s;
    s.resolution = 480;
    Note n;
    n.pitch = pitch;
    n.velocity = vel;
    n.onset_ticks = onset;
    n.duration_ticks = dur;
    s.notes.push_back(n);
    Bar b;
    b.note_indices.push_back(0);
    b.ts_numerator = 4;
    b.ts_denominator = 4;
    Track t;
    t.instrument = 0;
    t.type = TrackType::Melodic;
    t.bars.push_back(b);
    s.tracks.push_back(t);
    return s;
}

int count_type(const std::vector<int>& toks, const Vocabulary& vocab, TokenType t) {
    auto r = vocab.range(t);
    if (r.first == -1) return 0;
    int n = 0;
    for (int tok : toks) if (tok >= r.first && tok < r.second) ++n;
    return n;
}

}  // namespace

// ---------------------------------------------------------------------------
// Baseline single-track roundtrip
// ---------------------------------------------------------------------------

TEST_CASE("Encoder/Decoder: single-track single-note roundtrip") {
    auto cfg = std_config();
    Vocabulary vocab(cfg);
    Encoder enc(vocab);
    Decoder dec(vocab);

    auto score = one_bar_one_note(60, 64, 48, 48);
    auto tokens = enc.encode(score);
    CHECK(tokens.size() > 0);

    Score back = dec.decode(tokens);
    REQUIRE(back.tracks.size() == 1);
    REQUIRE(back.tracks[0].bars.size() == 1);
    REQUIRE(back.notes.size() == 1);
    CHECK(back.notes[0].pitch == 60);
    CHECK(back.notes[0].onset_ticks == 48);
    CHECK(back.notes[0].duration_ticks == 48);
    // velocity 64 → bin 16 → decoded back to ~66 (existing test pattern)
    CHECK(std::abs(back.notes[0].velocity - 64) < 8);
}

// ---------------------------------------------------------------------------
// Multi-bar single-track
// ---------------------------------------------------------------------------

TEST_CASE("Encoder/Decoder: 3-bar single track roundtrip") {
    auto cfg = std_config();
    Vocabulary vocab(cfg);
    Encoder enc(vocab);
    Decoder dec(vocab);

    Score s;
    s.resolution = 480;
    for (int i = 0; i < 3; ++i) {
        Note n;
        n.pitch = 60 + i; n.velocity = 64;
        n.onset_ticks = 0; n.duration_ticks = 240;
        s.notes.push_back(n);
    }
    Track t; t.instrument = 0; t.type = TrackType::Melodic;
    for (int i = 0; i < 3; ++i) {
        Bar b;
        b.ts_numerator = 4; b.ts_denominator = 4;
        b.note_indices.push_back(i);
        t.bars.push_back(b);
    }
    s.tracks.push_back(t);

    auto tokens = enc.encode(s);
    auto bar_count = count_type(tokens, vocab, TokenType::Bar);
    CHECK(bar_count == 3);

    Score back = dec.decode(tokens);
    REQUIRE(back.tracks.size() == 1);
    CHECK(back.tracks[0].bars.size() == 3);
    CHECK(back.notes.size() == 3);
}

// ---------------------------------------------------------------------------
// Multi-track
// ---------------------------------------------------------------------------

TEST_CASE("Encoder/Decoder: 2-track score roundtrip") {
    auto cfg = std_config();
    Vocabulary vocab(cfg);
    Encoder enc(vocab);
    Decoder dec(vocab);

    Score s; s.resolution = 480;
    Note n1{60, 64, 0, 240, 0}; Note n2{36, 100, 0, 240, 0};
    s.notes.push_back(n1); s.notes.push_back(n2);
    for (int i = 0; i < 2; ++i) {
        Track t;
        t.instrument = (i == 0) ? 0 : 33;
        t.type = TrackType::Melodic;
        Bar b; b.ts_numerator = 4; b.ts_denominator = 4;
        b.note_indices.push_back(i);
        t.bars.push_back(b);
        s.tracks.push_back(t);
    }
    auto tokens = enc.encode(s);
    CHECK(count_type(tokens, vocab, TokenType::Track) == 2);

    Score back = dec.decode(tokens);
    REQUIRE(back.tracks.size() == 2);
    CHECK(back.notes.size() == 2);
}

// ---------------------------------------------------------------------------
// Drum vs. melodic
// ---------------------------------------------------------------------------

TEST_CASE("Encoder: drum track omits NoteDuration tokens") {
    auto cfg = std_config();
    Vocabulary vocab(cfg);
    Encoder enc(vocab);

    Score s; s.resolution = 480;
    Note n; n.pitch = 36; n.velocity = 100; n.onset_ticks = 0; n.duration_ticks = 120;
    s.notes.push_back(n);
    Track t; t.instrument = 0; t.type = TrackType::Drum;
    Bar b; b.ts_numerator = 4; b.ts_denominator = 4; b.note_indices.push_back(0);
    t.bars.push_back(b); s.tracks.push_back(t);

    auto tokens = enc.encode(s);
    CHECK(count_type(tokens, vocab, TokenType::NoteDuration) == 0);
    CHECK(count_type(tokens, vocab, TokenType::NoteOnset) == 1);
}

TEST_CASE("Encoder: melodic track emits NoteDuration tokens") {
    auto cfg = std_config();
    Vocabulary vocab(cfg);
    Encoder enc(vocab);
    auto s = one_bar_one_note();
    auto tokens = enc.encode(s);
    CHECK(count_type(tokens, vocab, TokenType::NoteDuration) == 1);
}

// ---------------------------------------------------------------------------
// velocity_sticky on/off
// ---------------------------------------------------------------------------

TEST_CASE("Encoder velocity_sticky=true: identical velocities emit one VelocityLevel") {
    auto cfg = std_config();
    cfg.velocity_sticky = true;
    Vocabulary vocab(cfg);
    Encoder enc(vocab);

    Score s; s.resolution = 480;
    Note a{60, 64, 0, 120, 0}, b{62, 64, 120, 120, 0};
    s.notes.push_back(a); s.notes.push_back(b);
    Track t; t.instrument = 0; t.type = TrackType::Melodic;
    Bar bar; bar.ts_numerator = 4; bar.ts_denominator = 4;
    bar.note_indices = {0, 1};
    t.bars.push_back(bar); s.tracks.push_back(t);

    auto tokens = enc.encode(s);
    CHECK(count_type(tokens, vocab, TokenType::VelocityLevel) == 1);
}

TEST_CASE("Encoder velocity_sticky=false: each note emits its own VelocityLevel") {
    auto cfg = std_config();
    cfg.velocity_sticky = false;
    Vocabulary vocab(cfg);
    Encoder enc(vocab);

    Score s; s.resolution = 480;
    Note a{60, 64, 0, 120, 0}, b{62, 64, 120, 120, 0};
    s.notes.push_back(a); s.notes.push_back(b);
    Track t; t.instrument = 0; t.type = TrackType::Melodic;
    Bar bar; bar.ts_numerator = 4; bar.ts_denominator = 4;
    bar.note_indices = {0, 1};
    t.bars.push_back(bar); s.tracks.push_back(t);

    auto tokens = enc.encode(s);
    CHECK(count_type(tokens, vocab, TokenType::VelocityLevel) == 2);
}

// ---------------------------------------------------------------------------
// emit_delta_tokens
// ---------------------------------------------------------------------------

TEST_CASE("Encoder emit_delta_tokens=true: delta-bearing notes emit Delta tokens") {
    auto cfg = std_config(/*vel*/ true, /*dur*/ true, /*delta*/ true);
    cfg.emit_delta_tokens = true;
    Vocabulary vocab(cfg);
    Encoder enc(vocab);

    Score s; s.resolution = 480;
    Note n; n.pitch = 60; n.velocity = 64;
    n.onset_ticks = 48; n.duration_ticks = 96; n.delta = 5;
    s.notes.push_back(n);
    Track t; t.instrument = 0; t.type = TrackType::Melodic;
    Bar b; b.ts_numerator = 4; b.ts_denominator = 4; b.note_indices.push_back(0);
    t.bars.push_back(b); s.tracks.push_back(t);

    auto tokens = enc.encode(s);
    CHECK(count_type(tokens, vocab, TokenType::Delta) >= 1);
}

// ---------------------------------------------------------------------------
// Decoder edge: vocab missing VelocityLevel — encode/decode without it
// ---------------------------------------------------------------------------

TEST_CASE("Decoder: vocab without VelocityLevel doesn't crash, decoded velocity is some default") {
    auto cfg = std_config(/*vel*/ false);
    Vocabulary vocab(cfg);
    Encoder enc(vocab);
    Decoder dec(vocab);
    auto s = one_bar_one_note();
    auto tokens = enc.encode(s);
    REQUIRE_NOTHROW(dec.decode(tokens));
    Score back = dec.decode(tokens);
    REQUIRE(back.notes.size() == 1);
    // No assertion on velocity value — encoder cannot record it; just must not crash.
}

// ---------------------------------------------------------------------------
// multi_fill infill: FILL_IN_PLACEHOLDER + FILL_IN_START/END blocks
// ---------------------------------------------------------------------------

TEST_CASE("Encoder multi_fill: emits FillInPlaceholder for masked bars and FillIn blocks") {
    auto cfg = std_config(/*vel*/ true, /*dur*/ true, /*delta*/ false, /*fillin*/ true);
    cfg.supports_infill = true;
    Vocabulary vocab(cfg);
    Encoder enc(vocab);

    Score s; s.resolution = 480;
    for (int i = 0; i < 3; ++i) {
        Note n; n.pitch = 60 + i; n.velocity = 64;
        n.onset_ticks = 0; n.duration_ticks = 240;
        s.notes.push_back(n);
    }
    Track t; t.instrument = 0; t.type = TrackType::Melodic;
    for (int i = 0; i < 3; ++i) {
        Bar b; b.ts_numerator = 4; b.ts_denominator = 4;
        b.note_indices.push_back(i);
        t.bars.push_back(b);
    }
    s.tracks.push_back(t);

    EncodeOptions opt;
    opt.multi_fill = {{0, 1}};  // infill bar 1
    auto tokens = enc.encode(s, opt);
    CHECK(count_type(tokens, vocab, TokenType::FillInPlaceholder) == 1);
    CHECK(count_type(tokens, vocab, TokenType::FillInStart) >= 1);
    CHECK(count_type(tokens, vocab, TokenType::FillInEnd) >= 1);
}

// ---------------------------------------------------------------------------
// use_span_masks: encode_full returns hidden_spans
// ---------------------------------------------------------------------------

TEST_CASE("Encoder use_span_masks=true: hidden_spans cover future bars") {
    auto cfg = std_config();
    Vocabulary vocab(cfg);
    Encoder enc(vocab);

    Score s; s.resolution = 480;
    Note n{60, 64, 0, 240, 0}; s.notes.push_back(n);
    Track t; t.instrument = 0; t.type = TrackType::Melodic;
    {
        Bar b; b.ts_numerator = 4; b.ts_denominator = 4; b.note_indices.push_back(0);
        t.bars.push_back(b);
    }
    for (int i = 0; i < 2; ++i) {
        Bar b; b.ts_numerator = 4; b.ts_denominator = 4; b.future = true;
        t.bars.push_back(b);
    }
    s.tracks.push_back(t);

    EncodeOptions opt; opt.use_span_masks = true;
    auto result = enc.encode_full(s, opt);
    CHECK(result.tokens.size() > 0);
    CHECK(result.hidden_spans.size() == 2);
    for (auto [a, b] : result.hidden_spans) {
        CHECK(a < b);
        CHECK(b <= (int)result.tokens.size());
    }
}

// ---------------------------------------------------------------------------
// remove_future_bars: future bars omitted entirely
// ---------------------------------------------------------------------------

TEST_CASE("Encoder remove_future_bars=true: future bars emit no tokens") {
    auto cfg = std_config();
    Vocabulary vocab(cfg);
    Encoder enc(vocab);

    Score s; s.resolution = 480;
    Note n{60, 64, 0, 240, 0}; s.notes.push_back(n);
    Track t; t.instrument = 0; t.type = TrackType::Melodic;
    Bar b1; b1.ts_numerator = 4; b1.ts_denominator = 4; b1.note_indices.push_back(0);
    t.bars.push_back(b1);
    Bar b2; b2.ts_numerator = 4; b2.ts_denominator = 4; b2.future = true;
    t.bars.push_back(b2);
    s.tracks.push_back(t);

    EncodeOptions a; auto baseline = enc.encode(s, a);
    EncodeOptions b; b.remove_future_bars = true; auto pruned = enc.encode(s, b);
    CHECK(pruned.size() < baseline.size());
    CHECK(count_type(pruned, vocab, TokenType::Bar) == 1);
}

// ---------------------------------------------------------------------------
// Partial-encode (suffix-AR): omit TrackEnd so the model can continue
// ---------------------------------------------------------------------------

TEST_CASE("Encoder partial_encode: only emits a prefix of bars, no TrackEnd") {
    auto cfg = std_config();
    Vocabulary vocab(cfg);
    Encoder enc(vocab);

    Score s; s.resolution = 480;
    for (int i = 0; i < 3; ++i) {
        Note n; n.pitch = 60 + i; n.velocity = 64;
        n.onset_ticks = 0; n.duration_ticks = 240;
        s.notes.push_back(n);
    }
    Track t; t.instrument = 0; t.type = TrackType::Melodic;
    for (int i = 0; i < 3; ++i) {
        Bar b; b.ts_numerator = 4; b.ts_denominator = 4;
        b.note_indices.push_back(i);
        t.bars.push_back(b);
    }
    s.tracks.push_back(t);

    EncodeOptions opt;
    opt.partial_encode_track_index = 0;
    opt.partial_encode_track_bars = 2;
    auto tokens = enc.encode(s, opt);
    CHECK(count_type(tokens, vocab, TokenType::Bar) == 2);
    CHECK(count_type(tokens, vocab, TokenType::TrackEnd) == 0);
}
