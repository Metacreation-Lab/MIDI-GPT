// Canonical StepPlanner suite. Covers AR + infill pass interplay, window/
// context centering, multi-track tps, ignore mask, default bars_per_step.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "../../src/cpp/sampling/selection_mask.h"
#include "../../src/cpp/sampling/generation_step.h"
#include "../../src/cpp/sampling/step_planner.h"
#include "../../src/cpp/tokenizer/encoder_config.h"

using namespace midigpt::sampling;
using namespace midigpt::tokenizer;

// ---------------------------------------------------------------------------
// Baseline: single track, AR, 4 bars selected, model_dim=2 → 2 steps
// ---------------------------------------------------------------------------

TEST_CASE("StepPlanner: 1 track 4 bars, AR, model_dim=2 yields 2 steps") {
    SelectionMask m;
    m.selected = {{true, true, true, true}};
    m.autoregressive = {true};
    m.ignore = {false};
    EncoderConfig cfg; cfg.model_dim = 2;
    StepPlanner p(m, cfg);
    auto steps = p.plan();
    REQUIRE(steps.size() == 2);
    CHECK(steps[0].start_bar == 0);
    CHECK(steps[0].end_bar == 2);
    CHECK(steps[1].start_bar == 2);
    CHECK(steps[1].end_bar == 4);
}

TEST_CASE("StepPlanner: 1 track 4 bars, AR, model_dim=2, bars_per_step=1 yields 3 steps") {
    SelectionMask m;
    m.selected = {{true, true, true, true}};
    m.autoregressive = {true};
    m.ignore = {false};
    EncoderConfig cfg; cfg.model_dim = 2;
    StepPlanner p(m, cfg, 1, 1);
    auto steps = p.plan();
    REQUIRE(steps.size() == 3);
    CHECK(steps[0].start_bar == 0);
    CHECK(steps[0].end_bar == 2);
    REQUIRE(steps[0].bars_to_generate.size() == 2);
    CHECK(steps[1].start_bar == 1);
    CHECK(steps[1].end_bar == 3);
    REQUIRE(steps[1].bars_to_generate.size() == 1);
    CHECK(steps[2].start_bar == 2);
    CHECK(steps[2].end_bar == 4);
    REQUIRE(steps[2].bars_to_generate.size() == 1);
}

// ---------------------------------------------------------------------------
// bars_per_step=0 default to model_dim
// ---------------------------------------------------------------------------

TEST_CASE("StepPlanner: bars_per_step=0 defaults to model_dim") {
    SelectionMask m;
    m.selected = {{true, true, true, true}};
    m.autoregressive = {true};
    m.ignore = {false};
    EncoderConfig cfg; cfg.model_dim = 4;
    StepPlanner p(m, cfg, /*bars_per_step=*/0);
    auto steps = p.plan();
    REQUIRE(steps.size() == 1);
    CHECK(steps[0].start_bar == 0);
    CHECK(steps[0].end_bar == 4);
}

// ---------------------------------------------------------------------------
// AR + infill mix: AR steps come before infill steps
// ---------------------------------------------------------------------------

TEST_CASE("StepPlanner: AR steps emitted before infill steps") {
    SelectionMask m;
    m.selected = {{true, true}, {true, true}};
    m.autoregressive = {true, false};
    m.ignore = {false, false};
    EncoderConfig cfg; cfg.model_dim = 2;
    StepPlanner p(m, cfg);
    auto steps = p.plan();
    REQUIRE(steps.size() >= 2);
    // The first emitted step must be AR if any AR work exists
    CHECK(steps.front().is_autoregressive == true);
    // And there's at least one infill step
    bool has_infill = false;
    for (auto& s : steps) if (!s.is_autoregressive) { has_infill = true; break; }
    CHECK(has_infill);
}

// ---------------------------------------------------------------------------
// ignore mask
// ---------------------------------------------------------------------------

TEST_CASE("StepPlanner: ignored tracks excluded from steps") {
    SelectionMask m;
    m.selected = {{true, true}, {true, true}};
    m.autoregressive = {true, true};
    m.ignore = {false, true};
    EncoderConfig cfg; cfg.model_dim = 2;
    StepPlanner p(m, cfg);
    auto steps = p.plan();
    REQUIRE(steps.size() >= 1);
    for (auto& s : steps) {
        for (auto& [lt, lb, gt, gb] : s.bar_mapping) {
            CHECK(gt != 1);  // track 1 must never appear in any step's targets
        }
        for (auto& [t, b] : s.bars_to_generate) {
            CHECK(t != 1);
        }
    }
}

// ---------------------------------------------------------------------------
// Multi-track tracks_per_step > 1
// ---------------------------------------------------------------------------

TEST_CASE("StepPlanner: tracks_per_step=2 covers both tracks in a single step") {
    SelectionMask m;
    m.selected = {{true, true}, {true, true}};
    m.autoregressive = {true, true};
    m.ignore = {false, false};
    EncoderConfig cfg; cfg.model_dim = 2;
    StepPlanner p(m, cfg, /*bps=*/0, /*tps=*/2);
    auto steps = p.plan();
    REQUIRE(steps.size() >= 1);
    // The first AR step should include both tracks in its track_indices
    int max_tracks_in_step = 0;
    for (auto& s : steps) max_tracks_in_step =
        std::max<int>(max_tracks_in_step, s.track_indices.size());
    CHECK(max_tracks_in_step >= 2);
}

// ---------------------------------------------------------------------------
// Sparse infill: only one bar marked
// ---------------------------------------------------------------------------

TEST_CASE("StepPlanner: single-bar infill produces one step with that bar") {
    SelectionMask m;
    m.selected = {{false, true, false, false, false, false}};
    m.autoregressive = {false};
    m.ignore = {false};
    EncoderConfig cfg; cfg.model_dim = 4;
    StepPlanner p(m, cfg);
    auto steps = p.plan();
    REQUIRE(steps.size() == 1);
    CHECK(steps[0].is_autoregressive == false);
    CHECK(steps[0].bars_to_generate.size() == 1);
    CHECK(steps[0].bars_to_generate.count({0, 1}) == 1);
}

// ---------------------------------------------------------------------------
// Empty selection → no steps
// ---------------------------------------------------------------------------

TEST_CASE("StepPlanner: empty selection yields empty plan") {
    SelectionMask m;
    m.selected = {{false, false, false}};
    m.autoregressive = {false};
    m.ignore = {false};
    EncoderConfig cfg; cfg.model_dim = 2;
    StepPlanner p(m, cfg);
    auto steps = p.plan();
    CHECK(steps.empty());
}

TEST_CASE("StepPlanner: completely empty SelectionMask yields empty plan") {
    SelectionMask m;
    EncoderConfig cfg; cfg.model_dim = 2;
    StepPlanner p(m, cfg);
    auto steps = p.plan();
    CHECK(steps.empty());
}

// ---------------------------------------------------------------------------
// bar_mapping invariants
// ---------------------------------------------------------------------------

TEST_CASE("StepPlanner: bar_mapping entries reference valid global indices") {
    SelectionMask m;
    m.selected = {{true, true, true, true}};
    m.autoregressive = {true};
    m.ignore = {false};
    EncoderConfig cfg; cfg.model_dim = 2;
    StepPlanner p(m, cfg);
    auto steps = p.plan();
    for (auto& s : steps) {
        for (auto& [lt, lb, gt, gb] : s.bar_mapping) {
            CHECK(gt >= 0);
            CHECK(gb >= s.start_bar);
            CHECK(gb < s.end_bar);
            CHECK(lb >= 0);
            CHECK(lb < (s.end_bar - s.start_bar));
        }
    }
}
