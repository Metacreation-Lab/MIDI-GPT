#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "../../src/cpp/sampling/selection_mask.h"
#include "../../src/cpp/sampling/generation_step.h"
#include "../../src/cpp/sampling/step_planner.h"
#include "../../src/cpp/sampling/session_state.h"
#include "../../src/cpp/tokenizer/encoder_config.h"

using namespace midigpt::sampling;
using namespace midigpt::tokenizer;

TEST_CASE("StepPlanner basics") {
    SelectionMask mask;
    mask.selected = {{true, true, false, true}}; // Track 0
    mask.ignore = {false};
    
    EncoderConfig config;
    config.model_dim = 2; // 2 bars per step
    
    StepPlanner planner(mask, config);
    auto steps = planner.plan();
    
    REQUIRE(steps.size() == 2);
    CHECK(steps[0].start_bar == 0);
    CHECK(steps[0].end_bar == 2);
    CHECK(steps[0].bars_to_generate.size() == 2);
    
    CHECK(steps[1].start_bar == 2);
    CHECK(steps[1].end_bar == 4);
    CHECK(steps[1].bars_to_generate.size() == 1);
}

// Simple test for SessionState isn't fully set up since it needs Score, Encoder, Decoder, etc.
// But StepPlanner is the core sampling logic currently implemented.
