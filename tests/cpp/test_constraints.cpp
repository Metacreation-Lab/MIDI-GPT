// Canonical constraint suite: every masking primitive lives here.
//   - GrammarConstraint (state machine + flags: exact_bars, max_tracks,
//     require_notes, autoregressive, drum, set_fillin_drum, mask_track_*)
//   - AttributeValueConstraint
//   - BarAttributeValueConstraint (cursor)
//   - DensityConstraint (counter + reset)
//   - PolyphonyConstraint (counter + reset)
//   - ConstraintGraph composition (OR-masking, propagation)
//
// Tests are RAW: no production code is altered to make them pass — failures
// are signal.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "../../src/cpp/masking/constraint_graph.h"
#include "../../src/cpp/masking/grammar_constraint.h"
#include "../../src/cpp/masking/attribute_value_constraint.h"
#include "../../src/cpp/masking/bar_attribute_value_constraint.h"
#include "../../src/cpp/masking/density_constraint.h"
#include "../../src/cpp/masking/polyphony_constraint.h"
#include "../../src/cpp/tokenizer/vocabulary.h"
#include "../../src/cpp/tokenizer/encoder_config.h"

using namespace midigpt;
using namespace midigpt::tokenizer;
using namespace midigpt::masking;
// Explicitly shadow winnt.h's TokenType enum value (Windows-only collision).
using TokenType = midigpt::TokenType;

namespace {

// Full vocab covering every token type the grammar references.
EncoderConfig full_config() {
    EncoderConfig cfg;
    cfg.resolution = 12;
    auto push = [&](TokenType t, int sz) {
        cfg.token_domains.push_back({t, sz});
    };
    push(TokenType::PieceStart, 1);
    push(TokenType::NumBars, 16);
    push(TokenType::Track, 10);
    push(TokenType::Instrument, 128);
    push(TokenType::Bar, 1);
    push(TokenType::TimeSig, 8);
    push(TokenType::Tension, 10);
    push(TokenType::PitchClassSet, 4096);
    push(TokenType::MaskBar, 1);
    push(TokenType::TimeAbsolutePos, 48);
    push(TokenType::VelocityLevel, 32);
    push(TokenType::NoteOnset, 128);
    push(TokenType::NotePitch, 128);
    push(TokenType::NoteDuration, 128);
    push(TokenType::BarEnd, 1);
    push(TokenType::TrackEnd, 1);
    push(TokenType::PieceEnd, 1);
    push(TokenType::FillInStart, 1);
    push(TokenType::FillInEnd, 1);
    push(TokenType::FillInPlaceholder, 1);
    push(TokenType::Delta, 64);
    push(TokenType::DeltaDirection, 2);
    push(TokenType::NoteDensity, 8);
    push(TokenType::MinPolyphony, 8);
    push(TokenType::MaxPolyphony, 8);
    push(TokenType::MinNoteDuration, 8);
    push(TokenType::MaxNoteDuration, 8);
    push(TokenType::PitchRange, 8);
    return cfg;
}

int count_masked(const std::vector<bool>& m, std::pair<int,int> r) {
    if (r.first == -1) return 0;
    int n = 0;
    for (int i = r.first; i < r.second; ++i) if (m[i]) ++n;
    return n;
}
bool all_masked(const std::vector<bool>& m, std::pair<int,int> r) {
    if (r.first == -1) return true;
    for (int i = r.first; i < r.second; ++i) if (!m[i]) return false;
    return true;
}
bool none_masked(const std::vector<bool>& m, std::pair<int,int> r) {
    if (r.first == -1) return true;
    for (int i = r.first; i < r.second; ++i) if (m[i]) return false;
    return true;
}

}  // namespace

// ---------------------------------------------------------------------------
// GrammarConstraint — happy-path FSM (covers what the old basic test covered)
// ---------------------------------------------------------------------------

TEST_CASE("Grammar FSM: PieceStart allows Track, blocks notes/bar") {
    auto cfg = full_config();
    Vocabulary vocab(cfg);
    ConstraintGraph graph;
    graph.add_constraint(std::make_shared<GrammarConstraint>());

    auto m = graph.get_mask(vocab);
    CHECK(none_masked(m, vocab.range(TokenType::Track)));
    CHECK(none_masked(m, vocab.range(TokenType::NumBars)));
    CHECK(all_masked(m, vocab.range(TokenType::Bar)));
    CHECK(all_masked(m, vocab.range(TokenType::NoteOnset)));
    CHECK(all_masked(m, vocab.range(TokenType::PieceEnd)));
}

TEST_CASE("Grammar FSM: Track allows Bar + Instrument, blocks notes") {
    // With exact_bars set, TrackEnd is masked until bar_count_ reaches it.
    // Default mode (exact_bars=-1) intentionally allows empty tracks.
    auto cfg = full_config();
    Vocabulary vocab(cfg);
    ConstraintGraph graph;
    auto gram = std::make_shared<GrammarConstraint>();
    gram->set_exact_bars(4);
    graph.add_constraint(gram);
    graph.step(vocab.encode(TokenType::Track, 0), vocab);
    auto m = graph.get_mask(vocab);
    CHECK(none_masked(m, vocab.range(TokenType::Bar)));
    CHECK(none_masked(m, vocab.range(TokenType::Instrument)));
    CHECK(all_masked(m, vocab.range(TokenType::NoteOnset)));
    CHECK(all_masked(m, vocab.range(TokenType::TrackEnd)));
}

TEST_CASE("Grammar FSM: Bar allows TimeAbsolutePos and (direct) NoteOnset") {
    auto cfg = full_config();
    Vocabulary vocab(cfg);
    ConstraintGraph graph;
    graph.add_constraint(std::make_shared<GrammarConstraint>());
    graph.step(vocab.encode(TokenType::Track, 0), vocab);
    graph.step(vocab.encode(TokenType::Bar, 0), vocab);
    auto m = graph.get_mask(vocab);
    CHECK(none_masked(m, vocab.range(TokenType::TimeAbsolutePos)));
    CHECK(none_masked(m, vocab.range(TokenType::NoteOnset)));
    // require_notes_ defaults to true → BarEnd masked at empty bar
    CHECK(all_masked(m, vocab.range(TokenType::BarEnd)));
}

TEST_CASE("Grammar FSM: TimeAbsolutePos then NoteOnset allowed") {
    auto cfg = full_config();
    Vocabulary vocab(cfg);
    ConstraintGraph graph;
    graph.add_constraint(std::make_shared<GrammarConstraint>());
    graph.step(vocab.encode(TokenType::Track, 0), vocab);
    graph.step(vocab.encode(TokenType::Bar, 0), vocab);
    graph.step(vocab.encode(TokenType::TimeAbsolutePos, 0), vocab);
    auto m = graph.get_mask(vocab);
    CHECK(none_masked(m, vocab.range(TokenType::NoteOnset)));
}

// ---------------------------------------------------------------------------
// exact_bars flag
// ---------------------------------------------------------------------------

TEST_CASE("Grammar exact_bars: TrackEnd masked while bar_count<exact_bars") {
    auto cfg = full_config();
    Vocabulary vocab(cfg);
    auto g = std::make_shared<GrammarConstraint>();
    g->set_exact_bars(3);
    g->set_require_notes(false);

    ConstraintGraph graph;
    graph.add_constraint(g);
    graph.step(vocab.encode(TokenType::Track, 0), vocab);
    graph.step(vocab.encode(TokenType::Bar, 0), vocab);
    graph.step(vocab.encode(TokenType::BarEnd, 0), vocab);
    auto m = graph.get_mask(vocab);
    CHECK(none_masked(m, vocab.range(TokenType::Bar)));
    CHECK(all_masked(m, vocab.range(TokenType::TrackEnd)));
    CHECK(all_masked(m, vocab.range(TokenType::PieceEnd)));
}

TEST_CASE("Grammar exact_bars: Bar masked, TrackEnd allowed after reaching count") {
    auto cfg = full_config();
    Vocabulary vocab(cfg);
    auto g = std::make_shared<GrammarConstraint>();
    g->set_exact_bars(2);
    g->set_require_notes(false);

    ConstraintGraph graph;
    graph.add_constraint(g);
    graph.step(vocab.encode(TokenType::Track, 0), vocab);
    graph.step(vocab.encode(TokenType::Bar, 0), vocab);
    graph.step(vocab.encode(TokenType::BarEnd, 0), vocab);
    graph.step(vocab.encode(TokenType::Bar, 0), vocab);
    graph.step(vocab.encode(TokenType::BarEnd, 0), vocab);
    auto m = graph.get_mask(vocab);
    CHECK(all_masked(m, vocab.range(TokenType::Bar)));
    CHECK(none_masked(m, vocab.range(TokenType::TrackEnd)));
    CHECK(none_masked(m, vocab.range(TokenType::PieceEnd)));
}

TEST_CASE("Grammar exact_bars: counter resets per Track") {
    auto cfg = full_config();
    Vocabulary vocab(cfg);
    auto g = std::make_shared<GrammarConstraint>();
    g->set_exact_bars(2);
    g->set_require_notes(false);

    ConstraintGraph graph;
    graph.add_constraint(g);
    for (int i = 0; i < 2; ++i) {
        graph.step(vocab.encode(TokenType::Bar, 0), vocab);
        graph.step(vocab.encode(TokenType::BarEnd, 0), vocab);
    }
    graph.step(vocab.encode(TokenType::TrackEnd, 0), vocab);
    graph.step(vocab.encode(TokenType::Track, 0), vocab);
    auto m = graph.get_mask(vocab);
    CHECK(none_masked(m, vocab.range(TokenType::Bar)));
}

// ---------------------------------------------------------------------------
// max_tracks
// ---------------------------------------------------------------------------

TEST_CASE("Grammar max_tracks: new Track masked after reaching cap") {
    auto cfg = full_config();
    Vocabulary vocab(cfg);
    auto g = std::make_shared<GrammarConstraint>();
    g->set_max_tracks(2);
    g->set_require_notes(false);

    ConstraintGraph graph;
    graph.add_constraint(g);
    auto run = [&]() {
        graph.step(vocab.encode(TokenType::Track, 0), vocab);
        graph.step(vocab.encode(TokenType::Bar, 0), vocab);
        graph.step(vocab.encode(TokenType::BarEnd, 0), vocab);
        graph.step(vocab.encode(TokenType::TrackEnd, 0), vocab);
    };
    run();
    CHECK(none_masked(graph.get_mask(vocab), vocab.range(TokenType::Track)));
    run();
    auto m = graph.get_mask(vocab);
    CHECK(all_masked(m, vocab.range(TokenType::Track)));
    CHECK(none_masked(m, vocab.range(TokenType::PieceEnd)));
}

// ---------------------------------------------------------------------------
// require_notes
// ---------------------------------------------------------------------------

TEST_CASE("Grammar require_notes=true: BarEnd masked at empty bar") {
    auto cfg = full_config();
    Vocabulary vocab(cfg);
    auto g = std::make_shared<GrammarConstraint>();
    g->set_require_notes(true);
    ConstraintGraph graph; graph.add_constraint(g);
    graph.step(vocab.encode(TokenType::Track, 0), vocab);
    graph.step(vocab.encode(TokenType::Bar, 0), vocab);
    CHECK(all_masked(graph.get_mask(vocab), vocab.range(TokenType::BarEnd)));
}

TEST_CASE("Grammar require_notes=false: BarEnd allowed at empty bar") {
    auto cfg = full_config();
    Vocabulary vocab(cfg);
    auto g = std::make_shared<GrammarConstraint>();
    g->set_require_notes(false);
    ConstraintGraph graph; graph.add_constraint(g);
    graph.step(vocab.encode(TokenType::Track, 0), vocab);
    graph.step(vocab.encode(TokenType::Bar, 0), vocab);
    CHECK(none_masked(graph.get_mask(vocab), vocab.range(TokenType::BarEnd)));
}

TEST_CASE("Grammar require_notes=true: BarEnd allowed after NoteOnset+Duration") {
    auto cfg = full_config();
    Vocabulary vocab(cfg);
    auto g = std::make_shared<GrammarConstraint>();
    g->set_require_notes(true);
    ConstraintGraph graph; graph.add_constraint(g);
    graph.step(vocab.encode(TokenType::Track, 0), vocab);
    graph.step(vocab.encode(TokenType::Bar, 0), vocab);
    graph.step(vocab.encode(TokenType::TimeAbsolutePos, 0), vocab);
    graph.step(vocab.encode(TokenType::NoteOnset, 60), vocab);
    graph.step(vocab.encode(TokenType::NoteDuration, 4), vocab);
    CHECK(none_masked(graph.get_mask(vocab), vocab.range(TokenType::BarEnd)));
}

// ---------------------------------------------------------------------------
// is_autoregressive_ flag
// ---------------------------------------------------------------------------

TEST_CASE("Grammar AR mode: FillIn-* always masked across reachable states") {
    auto cfg = full_config();
    Vocabulary vocab(cfg);
    auto g = std::make_shared<GrammarConstraint>();
    g->set_autoregressive_mode(true);

    ConstraintGraph graph; graph.add_constraint(g);
    auto check = [&]() {
        auto m = graph.get_mask(vocab);
        CHECK(all_masked(m, vocab.range(TokenType::FillInStart)));
        CHECK(all_masked(m, vocab.range(TokenType::FillInEnd)));
        CHECK(all_masked(m, vocab.range(TokenType::FillInPlaceholder)));
    };
    check();
    graph.step(vocab.encode(TokenType::Track, 0), vocab); check();
    graph.step(vocab.encode(TokenType::Bar, 0), vocab);   check();
    graph.step(vocab.encode(TokenType::TimeAbsolutePos, 0), vocab); check();
    graph.step(vocab.encode(TokenType::NoteOnset, 60), vocab); check();
}

TEST_CASE("Grammar non-AR: FillInStart reachable from BarEnd") {
    auto cfg = full_config();
    Vocabulary vocab(cfg);
    auto g = std::make_shared<GrammarConstraint>();
    g->set_autoregressive_mode(false);
    g->set_require_notes(false);
    ConstraintGraph graph; graph.add_constraint(g);
    graph.step(vocab.encode(TokenType::Track, 0), vocab);
    graph.step(vocab.encode(TokenType::Bar, 0), vocab);
    graph.step(vocab.encode(TokenType::BarEnd, 0), vocab);
    CHECK(none_masked(graph.get_mask(vocab), vocab.range(TokenType::FillInStart)));
}

// ---------------------------------------------------------------------------
// Drum-flag derivation (Track value=1) + set_fillin_drum override
// ---------------------------------------------------------------------------

TEST_CASE("Grammar drum from Track value=1: NoteDuration not allowed after NoteOnset") {
    auto cfg = full_config();
    Vocabulary vocab(cfg);
    auto g = std::make_shared<GrammarConstraint>();
    g->set_require_notes(false);
    ConstraintGraph graph; graph.add_constraint(g);
    graph.step(vocab.encode(TokenType::Track, 1), vocab);  // drum
    graph.step(vocab.encode(TokenType::Bar, 0), vocab);
    graph.step(vocab.encode(TokenType::TimeAbsolutePos, 0), vocab);
    graph.step(vocab.encode(TokenType::NoteOnset, 36), vocab);
    auto m = graph.get_mask(vocab);
    CHECK(all_masked(m, vocab.range(TokenType::NoteDuration)));
    CHECK(none_masked(m, vocab.range(TokenType::NoteOnset)));
}

TEST_CASE("Grammar melodic from Track value=0: NoteDuration required after NoteOnset") {
    auto cfg = full_config();
    Vocabulary vocab(cfg);
    auto g = std::make_shared<GrammarConstraint>();
    g->set_require_notes(false);
    ConstraintGraph graph; graph.add_constraint(g);
    graph.step(vocab.encode(TokenType::Track, 0), vocab);
    graph.step(vocab.encode(TokenType::Bar, 0), vocab);
    graph.step(vocab.encode(TokenType::TimeAbsolutePos, 0), vocab);
    graph.step(vocab.encode(TokenType::NoteOnset, 60), vocab);
    auto m = graph.get_mask(vocab);
    CHECK(none_masked(m, vocab.range(TokenType::NoteDuration)));
    CHECK(all_masked(m, vocab.range(TokenType::NoteOnset)));
}

TEST_CASE("Grammar set_fillin_drum(true) overrides Track=0 melodic flag") {
    auto cfg = full_config();
    Vocabulary vocab(cfg);
    auto g = std::make_shared<GrammarConstraint>();
    g->set_require_notes(false);
    ConstraintGraph graph; graph.add_constraint(g);
    graph.step(vocab.encode(TokenType::Track, 0), vocab);
    graph.set_fillin_drum(true);
    graph.step(vocab.encode(TokenType::Bar, 0), vocab);
    graph.step(vocab.encode(TokenType::TimeAbsolutePos, 0), vocab);
    graph.step(vocab.encode(TokenType::NoteOnset, 36), vocab);
    CHECK(all_masked(graph.get_mask(vocab), vocab.range(TokenType::NoteDuration)));
}

// ---------------------------------------------------------------------------
// mask_track_start / mask_track_end
// ---------------------------------------------------------------------------

TEST_CASE("Grammar mask_track_start: blocks Track tokens") {
    auto cfg = full_config();
    Vocabulary vocab(cfg);
    auto g = std::make_shared<GrammarConstraint>();
    g->set_mask_track_start(true);
    ConstraintGraph graph; graph.add_constraint(g);
    CHECK(all_masked(graph.get_mask(vocab), vocab.range(TokenType::Track)));
}

TEST_CASE("Grammar mask_track_end: blocks TrackEnd even when reachable") {
    auto cfg = full_config();
    Vocabulary vocab(cfg);
    auto g = std::make_shared<GrammarConstraint>();
    g->set_mask_track_end(true);
    g->set_require_notes(false);
    ConstraintGraph graph; graph.add_constraint(g);
    graph.step(vocab.encode(TokenType::Track, 0), vocab);
    graph.step(vocab.encode(TokenType::Bar, 0), vocab);
    graph.step(vocab.encode(TokenType::BarEnd, 0), vocab);
    CHECK(all_masked(graph.get_mask(vocab), vocab.range(TokenType::TrackEnd)));
}

// ---------------------------------------------------------------------------
// TimeAbsolutePos monotonicity + bar-length boundary
// ---------------------------------------------------------------------------

TEST_CASE("Grammar TimeAbsolutePos: pos<=current and pos>=bar_ticks both masked") {
    auto cfg = full_config();
    Vocabulary vocab(cfg);
    auto g = std::make_shared<GrammarConstraint>();
    g->set_require_notes(false);
    ConstraintGraph graph; graph.add_constraint(g);
    graph.step(vocab.encode(TokenType::Track, 0), vocab);
    graph.step(vocab.encode(TokenType::Bar, 0), vocab);
    graph.step(vocab.encode(TokenType::TimeAbsolutePos, 10), vocab);
    graph.step(vocab.encode(TokenType::NoteOnset, 60), vocab);
    graph.step(vocab.encode(TokenType::NoteDuration, 4), vocab);
    auto m = graph.get_mask(vocab);
    auto [s, e] = vocab.range(TokenType::TimeAbsolutePos);
    REQUIRE(s != -1);
    int bar_ticks = 4 * cfg.resolution;
    for (int i = s; i < e; ++i) {
        auto [_t, pos] = vocab.decode(i);
        if (pos <= 10 || pos >= bar_ticks) {
            CHECK(m[i] == true);
        } else {
            CHECK(m[i] == false);
        }
    }
}

// ---------------------------------------------------------------------------
// AttributeValueConstraint
// ---------------------------------------------------------------------------

TEST_CASE("AttributeValueConstraint: masks all but allowed value") {
    auto cfg = full_config();
    Vocabulary vocab(cfg);
    ConstraintGraph g;
    g.add_constraint(std::make_shared<AttributeValueConstraint>(
        TokenType::NoteOnset, 60));
    auto m = g.get_mask(vocab);
    auto [s, e] = vocab.range(TokenType::NoteOnset);
    int n = e - s;
    CHECK(count_masked(m, {s, e}) == n - 1);
    CHECK(m[s + 60] == false);
}

TEST_CASE("AttributeValueConstraint: target type not in vocab is silent no-op") {
    EncoderConfig cfg;
    cfg.resolution = 12;
    cfg.token_domains.push_back({TokenType::Track, 5});
    Vocabulary vocab(cfg);
    AttributeValueConstraint c(TokenType::Tension, 3);
    std::vector<bool> m(vocab.size(), false);
    REQUIRE_NOTHROW(c.apply(m, vocab));
    for (auto b : m) CHECK(b == false);
}

TEST_CASE("AttributeValueConstraint: leaves non-target ranges untouched") {
    auto cfg = full_config();
    Vocabulary vocab(cfg);
    ConstraintGraph g;
    g.add_constraint(std::make_shared<AttributeValueConstraint>(
        TokenType::Tension, 4));
    auto m = g.get_mask(vocab);
    CHECK(count_masked(m, vocab.range(TokenType::NoteOnset)) == 0);
    CHECK(count_masked(m, vocab.range(TokenType::Track))     == 0);
}

// ---------------------------------------------------------------------------
// BarAttributeValueConstraint
// ---------------------------------------------------------------------------

TEST_CASE("BarAttributeValueConstraint: no-op before any Track/Bar") {
    auto cfg = full_config();
    Vocabulary vocab(cfg);
    BarAttributeValueConstraint c(TokenType::Tension, 0, 0, 5);
    std::vector<bool> m(vocab.size(), false);
    c.apply(m, vocab);
    CHECK(count_masked(m, vocab.range(TokenType::Tension)) == 0);
}

TEST_CASE("BarAttributeValueConstraint: fires at (track=0, bar=0)") {
    auto cfg = full_config();
    Vocabulary vocab(cfg);
    BarAttributeValueConstraint c(TokenType::Tension, 0, 0, 5);
    c.step(vocab.encode(TokenType::Track, 0), vocab);
    c.step(vocab.encode(TokenType::Bar, 0), vocab);
    std::vector<bool> m(vocab.size(), false);
    c.apply(m, vocab);
    auto r = vocab.range(TokenType::Tension);
    int n = r.second - r.first;
    CHECK(count_masked(m, r) == n - 1);
    CHECK(m[r.first + 5] == false);
}

TEST_CASE("BarAttributeValueConstraint: no-op past target bar") {
    auto cfg = full_config();
    Vocabulary vocab(cfg);
    BarAttributeValueConstraint c(TokenType::Tension, 0, 0, 5);
    c.step(vocab.encode(TokenType::Track, 0), vocab);
    c.step(vocab.encode(TokenType::Bar, 0), vocab);
    c.step(vocab.encode(TokenType::Bar, 0), vocab);  // advance past bar 0
    std::vector<bool> m(vocab.size(), false);
    c.apply(m, vocab);
    CHECK(count_masked(m, vocab.range(TokenType::Tension)) == 0);
}

TEST_CASE("BarAttributeValueConstraint: fires at (track=1, bar=2)") {
    auto cfg = full_config();
    Vocabulary vocab(cfg);
    BarAttributeValueConstraint c(TokenType::Tension, 1, 2, 7);
    c.step(vocab.encode(TokenType::Track, 0), vocab);
    for (int i = 0; i < 3; ++i) c.step(vocab.encode(TokenType::Bar, 0), vocab);
    c.step(vocab.encode(TokenType::Track, 0), vocab);
    for (int i = 0; i < 3; ++i) c.step(vocab.encode(TokenType::Bar, 0), vocab);
    std::vector<bool> m(vocab.size(), false);
    c.apply(m, vocab);
    auto r = vocab.range(TokenType::Tension);
    int n = r.second - r.first;
    CHECK(count_masked(m, r) == n - 1);
    CHECK(m[r.first + 7] == false);
}

TEST_CASE("BarAttributeValueConstraint: wrong track is no-op") {
    auto cfg = full_config();
    Vocabulary vocab(cfg);
    BarAttributeValueConstraint c(TokenType::Tension, 2, 0, 1);
    c.step(vocab.encode(TokenType::Track, 0), vocab);
    c.step(vocab.encode(TokenType::Bar, 0), vocab);
    std::vector<bool> m(vocab.size(), false);
    c.apply(m, vocab);
    CHECK(count_masked(m, vocab.range(TokenType::Tension)) == 0);
}

TEST_CASE("BarAttributeValueConstraint: target type not in vocab is silent no-op") {
    EncoderConfig cfg;
    cfg.resolution = 12;
    cfg.token_domains.push_back({TokenType::Track, 5});
    cfg.token_domains.push_back({TokenType::Bar, 1});
    Vocabulary vocab(cfg);
    BarAttributeValueConstraint c(TokenType::Tension, 0, 0, 5);
    c.step(vocab.encode(TokenType::Track, 0), vocab);
    c.step(vocab.encode(TokenType::Bar, 0), vocab);
    std::vector<bool> m(vocab.size(), false);
    REQUIRE_NOTHROW(c.apply(m, vocab));
    for (auto b : m) CHECK(b == false);
}

TEST_CASE("BarAttributeValueConstraint: out-of-range allowed_value masks ALL") {
    auto cfg = full_config();
    Vocabulary vocab(cfg);
    auto r = vocab.range(TokenType::Tension);
    int n = r.second - r.first;
    BarAttributeValueConstraint c(TokenType::Tension, 0, 0, n + 100);
    c.step(vocab.encode(TokenType::Track, 0), vocab);
    c.step(vocab.encode(TokenType::Bar, 0), vocab);
    std::vector<bool> m(vocab.size(), false);
    c.apply(m, vocab);
    CHECK(count_masked(m, r) == n);
}

TEST_CASE("BarAttributeValueConstraint: boundary values 0 and size-1") {
    auto cfg = full_config();
    Vocabulary vocab(cfg);
    auto r = vocab.range(TokenType::Tension);
    int n = r.second - r.first;
    for (int v : {0, n - 1}) {
        BarAttributeValueConstraint c(TokenType::Tension, 0, 0, v);
        c.step(vocab.encode(TokenType::Track, 0), vocab);
        c.step(vocab.encode(TokenType::Bar, 0), vocab);
        std::vector<bool> m(vocab.size(), false);
        c.apply(m, vocab);
        CHECK(m[r.first + v] == false);
        CHECK(count_masked(m, r) == n - 1);
    }
}

TEST_CASE("BarAttributeValueConstraint: multiple stack on different bars") {
    auto cfg = full_config();
    Vocabulary vocab(cfg);
    ConstraintGraph g;
    g.add_constraint(std::make_shared<BarAttributeValueConstraint>(
        TokenType::Tension, 0, 0, 1));
    g.add_constraint(std::make_shared<BarAttributeValueConstraint>(
        TokenType::Tension, 0, 2, 8));
    g.step(vocab.encode(TokenType::Track, 0), vocab);
    g.step(vocab.encode(TokenType::Bar, 0), vocab);
    auto r = vocab.range(TokenType::Tension);
    int n = r.second - r.first;
    auto m0 = g.get_mask(vocab);
    CHECK(m0[r.first + 1] == false);
    CHECK(count_masked(m0, r) == n - 1);
    g.step(vocab.encode(TokenType::Bar, 0), vocab);  // bar 1
    auto m1 = g.get_mask(vocab);
    CHECK(count_masked(m1, r) == 0);
    g.step(vocab.encode(TokenType::Bar, 0), vocab);  // bar 2
    auto m2 = g.get_mask(vocab);
    CHECK(m2[r.first + 8] == false);
    CHECK(count_masked(m2, r) == n - 1);
}

// ---------------------------------------------------------------------------
// DensityConstraint
// ---------------------------------------------------------------------------

TEST_CASE("DensityConstraint: blocks notes/time after reaching count") {
    auto cfg = full_config();
    Vocabulary vocab(cfg);
    DensityConstraint c(2);
    c.step(vocab.encode(TokenType::Bar, 0), vocab);
    c.step(vocab.encode(TokenType::TimeAbsolutePos, 0), vocab);
    c.step(vocab.encode(TokenType::NoteOnset, 60), vocab);
    std::vector<bool> m1(vocab.size(), false);
    c.apply(m1, vocab);
    CHECK(count_masked(m1, vocab.range(TokenType::NoteOnset)) == 0);

    c.step(vocab.encode(TokenType::TimeAbsolutePos, 4), vocab);
    c.step(vocab.encode(TokenType::NoteOnset, 62), vocab);
    std::vector<bool> m2(vocab.size(), false);
    c.apply(m2, vocab);
    CHECK(all_masked(m2, vocab.range(TokenType::NoteOnset)));
    CHECK(all_masked(m2, vocab.range(TokenType::NotePitch)));
    CHECK(all_masked(m2, vocab.range(TokenType::VelocityLevel)));
    CHECK(all_masked(m2, vocab.range(TokenType::TimeAbsolutePos)));
    CHECK(all_masked(m2, vocab.range(TokenType::Delta)));
    CHECK(all_masked(m2, vocab.range(TokenType::DeltaDirection)));
}

TEST_CASE("DensityConstraint: counter resets on Bar / BarEnd") {
    auto cfg = full_config();
    Vocabulary vocab(cfg);
    DensityConstraint c(1);
    c.step(vocab.encode(TokenType::Bar, 0), vocab);
    c.step(vocab.encode(TokenType::NoteOnset, 60), vocab);
    std::vector<bool> mm(vocab.size(), false);
    c.apply(mm, vocab);
    CHECK(all_masked(mm, vocab.range(TokenType::NoteOnset)));

    c.step(vocab.encode(TokenType::BarEnd, 0), vocab);
    std::vector<bool> after(vocab.size(), false);
    c.apply(after, vocab);
    CHECK(count_masked(after, vocab.range(TokenType::NoteOnset)) == 0);
}

// ---------------------------------------------------------------------------
// PolyphonyConstraint
// ---------------------------------------------------------------------------

TEST_CASE("PolyphonyConstraint: blocks more notes at same timestep after max") {
    auto cfg = full_config();
    Vocabulary vocab(cfg);
    PolyphonyConstraint c(2);
    c.step(vocab.encode(TokenType::Bar, 0), vocab);
    c.step(vocab.encode(TokenType::TimeAbsolutePos, 0), vocab);
    c.step(vocab.encode(TokenType::NoteOnset, 60), vocab);
    c.step(vocab.encode(TokenType::NoteOnset, 64), vocab);
    std::vector<bool> m(vocab.size(), false);
    c.apply(m, vocab);
    CHECK(all_masked(m, vocab.range(TokenType::NoteOnset)));
    CHECK(all_masked(m, vocab.range(TokenType::NotePitch)));
}

TEST_CASE("PolyphonyConstraint: new TimeAbsolutePos resets counter") {
    auto cfg = full_config();
    Vocabulary vocab(cfg);
    PolyphonyConstraint c(2);
    c.step(vocab.encode(TokenType::Bar, 0), vocab);
    c.step(vocab.encode(TokenType::TimeAbsolutePos, 0), vocab);
    c.step(vocab.encode(TokenType::NoteOnset, 60), vocab);
    c.step(vocab.encode(TokenType::NoteOnset, 64), vocab);
    c.step(vocab.encode(TokenType::TimeAbsolutePos, 4), vocab);
    std::vector<bool> m(vocab.size(), false);
    c.apply(m, vocab);
    CHECK(count_masked(m, vocab.range(TokenType::NoteOnset)) == 0);
}

TEST_CASE("PolyphonyConstraint: Bar/BarEnd also reset counter") {
    auto cfg = full_config();
    Vocabulary vocab(cfg);
    PolyphonyConstraint c(1);
    c.step(vocab.encode(TokenType::TimeAbsolutePos, 0), vocab);
    c.step(vocab.encode(TokenType::NoteOnset, 60), vocab);
    std::vector<bool> blocked(vocab.size(), false);
    c.apply(blocked, vocab);
    CHECK(all_masked(blocked, vocab.range(TokenType::NoteOnset)));
    c.step(vocab.encode(TokenType::BarEnd, 0), vocab);
    std::vector<bool> reset(vocab.size(), false);
    c.apply(reset, vocab);
    CHECK(count_masked(reset, vocab.range(TokenType::NoteOnset)) == 0);
}

// ---------------------------------------------------------------------------
// ConstraintGraph composition (OR-mask, step/set_fillin_drum propagation)
// ---------------------------------------------------------------------------

TEST_CASE("ConstraintGraph: empty graph leaves mask all-false") {
    auto cfg = full_config();
    Vocabulary vocab(cfg);
    ConstraintGraph g;
    auto m = g.get_mask(vocab);
    REQUIRE(m.size() == (size_t)vocab.size());
    for (auto b : m) CHECK(b == false);
}

TEST_CASE("ConstraintGraph: multiple constraints OR-combine masks") {
    auto cfg = full_config();
    Vocabulary vocab(cfg);
    ConstraintGraph g;
    g.add_constraint(std::make_shared<GrammarConstraint>());
    g.add_constraint(std::make_shared<AttributeValueConstraint>(
        TokenType::Track, 3));
    auto m = g.get_mask(vocab);
    auto r = vocab.range(TokenType::Track);
    int n = r.second - r.first;
    // Grammar would allow all Track values at PieceStart; AVC additionally
    // restricts to value=3. The combined mask must mask everything but value=3.
    CHECK(m[r.first + 3] == false);
    CHECK(count_masked(m, r) == n - 1);
}

TEST_CASE("ConstraintGraph: step propagates to all constraints") {
    auto cfg = full_config();
    Vocabulary vocab(cfg);
    auto bar_attr = std::make_shared<BarAttributeValueConstraint>(
        TokenType::Tension, 0, 0, 5);
    auto density = std::make_shared<DensityConstraint>(2);

    ConstraintGraph g;
    g.add_constraint(bar_attr);
    g.add_constraint(density);

    g.step(vocab.encode(TokenType::Track, 0), vocab);
    g.step(vocab.encode(TokenType::Bar, 0), vocab);
    // bar_attr should now fire at (0,0); density counter still 0.
    auto m = g.get_mask(vocab);
    auto r = vocab.range(TokenType::Tension);
    int n = r.second - r.first;
    CHECK(m[r.first + 5] == false);
    CHECK(count_masked(m, r) == n - 1);
    CHECK(count_masked(m, vocab.range(TokenType::NoteOnset)) == 0);
}

TEST_CASE("ConstraintGraph: set_fillin_drum propagates to GrammarConstraint") {
    auto cfg = full_config();
    Vocabulary vocab(cfg);
    auto g_grammar = std::make_shared<GrammarConstraint>();
    g_grammar->set_require_notes(false);

    ConstraintGraph g; g.add_constraint(g_grammar);
    g.step(vocab.encode(TokenType::Track, 0), vocab);  // melodic
    g.set_fillin_drum(true);                            // override
    g.step(vocab.encode(TokenType::Bar, 0), vocab);
    g.step(vocab.encode(TokenType::TimeAbsolutePos, 0), vocab);
    g.step(vocab.encode(TokenType::NoteOnset, 36), vocab);
    CHECK(all_masked(g.get_mask(vocab), vocab.range(TokenType::NoteDuration)));
}

TEST_CASE("ConstraintGraph: null constraint is ignored, not crashed") {
    auto cfg = full_config();
    Vocabulary vocab(cfg);
    ConstraintGraph g;
    REQUIRE_NOTHROW(g.add_constraint(nullptr));
    REQUIRE_NOTHROW(g.step(vocab.encode(TokenType::Track, 0), vocab));
    REQUIRE_NOTHROW(g.get_mask(vocab));
}
