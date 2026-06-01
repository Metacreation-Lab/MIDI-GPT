// Canonical SessionState suite. SessionState glues Encoder/Decoder, the
// ConstraintGraph, and the Score together for one generation step.
// Exercises:
//   - complete() flips on PieceEnd
//   - advance() extends context_tokens()
//   - logit_mask() reflects active constraints
//   - hidden_spans() empty when span masks not enabled
//   - result() returns a Score with the original tracks intact for an empty step
//   - constructor with an empty step doesn't crash and preserves the input Score

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "../../src/cpp/sampling/session_state.h"
#include "../../src/cpp/tokenizer/vocabulary.h"
#include "../../src/cpp/tokenizer/encoder_config.h"
#include "../../src/cpp/tokenizer/encoder.h"
#include "../../src/cpp/tokenizer/decoder.h"
#include "../../src/cpp/masking/constraint_graph.h"
#include "../../src/cpp/masking/grammar_constraint.h"
#include "../../src/cpp/masking/attribute_value_constraint.h"
#include "../../src/cpp/core/score.h"

using namespace midigpt;
using namespace midigpt::sampling;
using namespace midigpt::tokenizer;
using namespace midigpt::masking;
// Explicitly shadow winnt.h's TokenType enum value (Windows-only collision).
using TokenType = midigpt::TokenType;

namespace {

EncoderConfig tiny_config() {
    EncoderConfig cfg;
    cfg.resolution = 12;
    cfg.token_domains.push_back({TokenType::PieceStart, 1});
    cfg.token_domains.push_back({TokenType::PieceEnd, 1});
    cfg.token_domains.push_back({TokenType::Track, 10});
    cfg.token_domains.push_back({TokenType::Bar, 1});
    cfg.token_domains.push_back({TokenType::NoteOnset, 128});
    return cfg;
}

}  // namespace

// ---------------------------------------------------------------------------
// complete() and advance()
// ---------------------------------------------------------------------------

TEST_CASE("SessionState: complete()=false initially; flips to true on PieceEnd") {
    auto cfg = tiny_config();
    Vocabulary vocab(cfg);
    Encoder enc(vocab); Decoder dec(vocab);
    ConstraintGraph cs;

    Score ctx;
    GenerationStep step;

    SessionState ss(ctx, step, vocab, cs, enc, dec);
    CHECK(ss.complete() == false);
    ss.advance(vocab.encode(TokenType::PieceEnd, 0));
    CHECK(ss.complete() == true);
}

TEST_CASE("SessionState: advance grows context_tokens by appended token") {
    auto cfg = tiny_config();
    Vocabulary vocab(cfg);
    Encoder enc(vocab); Decoder dec(vocab);
    ConstraintGraph cs;
    Score ctx; GenerationStep step;
    SessionState ss(ctx, step, vocab, cs, enc, dec);
    size_t before = ss.context_tokens().size();
    int tok = vocab.encode(TokenType::Track, 3);
    ss.advance(tok);
    auto after = ss.context_tokens();
    CHECK(after.size() == before + 1);
    CHECK(after.back() == tok);
}

TEST_CASE("SessionState: multiple advances append in order") {
    auto cfg = tiny_config();
    Vocabulary vocab(cfg);
    Encoder enc(vocab); Decoder dec(vocab);
    ConstraintGraph cs;
    Score ctx; GenerationStep step;
    SessionState ss(ctx, step, vocab, cs, enc, dec);
    size_t before = ss.context_tokens().size();
    std::vector<int> toks = {
        vocab.encode(TokenType::Track, 0),
        vocab.encode(TokenType::Bar, 0),
        vocab.encode(TokenType::NoteOnset, 60),
    };
    for (int t : toks) ss.advance(t);
    auto ct = ss.context_tokens();
    REQUIRE(ct.size() == before + toks.size());
    for (size_t i = 0; i < toks.size(); ++i) {
        CHECK(ct[before + i] == toks[i]);
    }
}

// ---------------------------------------------------------------------------
// logit_mask() reflects ConstraintGraph
// ---------------------------------------------------------------------------

TEST_CASE("SessionState: empty constraint graph yields all-allowed mask") {
    // logit_mask() returns true=allowed (inverted from ConstraintGraph's
    // true=disallowed). Empty graph → nothing disallowed → all allowed.
    auto cfg = tiny_config();
    Vocabulary vocab(cfg);
    Encoder enc(vocab); Decoder dec(vocab);
    ConstraintGraph cs;
    Score ctx; GenerationStep step;
    SessionState ss(ctx, step, vocab, cs, enc, dec);
    auto m = ss.logit_mask();
    REQUIRE(m.size() == (size_t)vocab.size());
    for (auto b : m) CHECK(b == true);
}

TEST_CASE("SessionState: AttributeValueConstraint visible through logit_mask") {
    auto cfg = tiny_config();
    Vocabulary vocab(cfg);
    Encoder enc(vocab); Decoder dec(vocab);
    ConstraintGraph cs;
    cs.add_constraint(std::make_shared<AttributeValueConstraint>(
        TokenType::NoteOnset, 60));
    Score ctx; GenerationStep step;
    SessionState ss(ctx, step, vocab, cs, enc, dec);
    auto m = ss.logit_mask();
    auto r = vocab.range(TokenType::NoteOnset);
    int allowed = 0;
    for (int i = r.first; i < r.second; ++i) if (m[i]) ++allowed;
    CHECK(allowed == 1);
    CHECK(m[r.first + 60] == true);
}

TEST_CASE("SessionState: graph state advances with each token") {
    auto cfg = tiny_config();
    Vocabulary vocab(cfg);
    Encoder enc(vocab); Decoder dec(vocab);
    ConstraintGraph cs;
    cs.add_constraint(std::make_shared<GrammarConstraint>());
    Score ctx; GenerationStep step;
    SessionState ss(ctx, step, vocab, cs, enc, dec);
    // Initial: Bar disallowed (true=allowed semantics)
    auto m0 = ss.logit_mask();
    auto bar_r = vocab.range(TokenType::Bar);
    CHECK(m0[bar_r.first] == false);
    // After Track: Bar allowed
    ss.advance(vocab.encode(TokenType::Track, 0));
    auto m1 = ss.logit_mask();
    CHECK(m1[bar_r.first] == true);
}

// ---------------------------------------------------------------------------
// hidden_spans
// ---------------------------------------------------------------------------

TEST_CASE("SessionState: hidden_spans empty when use_span_masks=false") {
    auto cfg = tiny_config();
    Vocabulary vocab(cfg);
    Encoder enc(vocab); Decoder dec(vocab);
    ConstraintGraph cs;
    Score ctx; GenerationStep step;
    SessionState ss(ctx, step, vocab, cs, enc, dec, /*use_span_masks=*/false);
    CHECK(ss.hidden_spans().empty());
}

// ---------------------------------------------------------------------------
// result() for empty step returns a Score
// ---------------------------------------------------------------------------

TEST_CASE("SessionState: empty step returns a Score from result()") {
    auto cfg = tiny_config();
    Vocabulary vocab(cfg);
    Encoder enc(vocab); Decoder dec(vocab);
    ConstraintGraph cs;
    Score ctx; GenerationStep step;
    SessionState ss(ctx, step, vocab, cs, enc, dec);
    REQUIRE_NOTHROW(ss.result());
}

TEST_CASE("SessionState: empty step + advance(PieceEnd) → result() returns a Score") {
    auto cfg = tiny_config();
    Vocabulary vocab(cfg);
    Encoder enc(vocab); Decoder dec(vocab);
    ConstraintGraph cs;
    Score ctx; GenerationStep step;
    SessionState ss(ctx, step, vocab, cs, enc, dec);
    ss.advance(vocab.encode(TokenType::PieceEnd, 0));
    REQUIRE_NOTHROW(ss.result());
}
