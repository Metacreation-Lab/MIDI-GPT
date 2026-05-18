#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "../../src/cpp/sampling/session_state.h"
#include "../../src/cpp/tokenizer/vocabulary.h"
#include "../../src/cpp/tokenizer/encoder_config.h"
#include "../../src/cpp/tokenizer/encoder.h"
#include "../../src/cpp/tokenizer/decoder.h"
#include "../../src/cpp/masking/constraint_graph.h"
#include "../../src/cpp/core/score.h"

using namespace midigpt;
using namespace midigpt::sampling;
using namespace midigpt::tokenizer;
using namespace midigpt::masking;

TEST_CASE("SessionState basics") {
    EncoderConfig config;
    // Set up a tiny vocab so we can test advance()
    config.token_domains.push_back({TokenType::PieceEnd, 1});
    config.token_domains.push_back({TokenType::NoteOnset, 128});
    Vocabulary vocab(config);
    
    Encoder encoder(vocab);
    Decoder decoder(vocab);
    ConstraintGraph constraints;
    
    Score context;
    GenerationStep step; // empty step
    
    SessionState session(context, step, vocab, constraints, encoder, decoder);
    
    CHECK(session.complete() == false);
    
    int piece_end_token = vocab.encode(TokenType::PieceEnd, 0);
    session.advance(piece_end_token);
    
    CHECK(session.complete() == true);
    
    auto tokens = session.context_tokens();
    // Context PieceEnd is stripped for autoregressive continuation, so only 1 generated token
    CHECK(tokens.size() == 1);
    CHECK(tokens[0] == piece_end_token);
}
