#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "../../src/cpp/masking/constraint_graph.h"
#include "../../src/cpp/masking/grammar_constraint.h"
#include "../../src/cpp/masking/attribute_value_constraint.h"

using namespace midigpt;
using namespace midigpt::tokenizer;
using namespace midigpt::masking;

TEST_CASE("ConstraintGraph and GrammarConstraint") {
    EncoderConfig config;
    config.token_domains.push_back({TokenType::PieceStart, 1});
    config.token_domains.push_back({TokenType::Track, 10});
    config.token_domains.push_back({TokenType::Bar, 1});
    config.token_domains.push_back({TokenType::TimeAbsolutePos, 100});
    config.token_domains.push_back({TokenType::NoteOnset, 128});
    config.token_domains.push_back({TokenType::NoteDuration, 128});
    config.token_domains.push_back({TokenType::BarEnd, 1});
    config.token_domains.push_back({TokenType::TrackEnd, 1});
    config.token_domains.push_back({TokenType::PieceEnd, 1});
    
    Vocabulary vocab(config);
    ConstraintGraph graph;
    
    graph.add_constraint(std::make_shared<GrammarConstraint>());
    
    // Initial state: PieceStart
    auto mask = graph.get_mask(vocab);
    
    // Track should be allowed, PieceEnd should be disallowed initially
    CHECK(mask[vocab.encode(TokenType::Track, 0)] == false);
    CHECK(mask[vocab.encode(TokenType::PieceEnd, 0)] == true);
    
    // Bar should be disallowed (mask = true)
    CHECK(mask[vocab.encode(TokenType::Bar, 0)] == true);
    CHECK(mask[vocab.encode(TokenType::NoteOnset, 0)] == true);
    
    // Step with Track
    graph.step(vocab.encode(TokenType::Track, 5), vocab);
    mask = graph.get_mask(vocab);
    
    // Now Bar is allowed, TrackEnd is disallowed because max_bars_ < 0 (requires Bar)
    CHECK(mask[vocab.encode(TokenType::Bar, 0)] == false);
    CHECK(mask[vocab.encode(TokenType::TrackEnd, 0)] == true);
    CHECK(mask[vocab.encode(TokenType::NoteOnset, 0)] == true);
    
    // Step with Bar
    graph.step(vocab.encode(TokenType::Bar, 0), vocab);
    mask = graph.get_mask(vocab);
    
    // Now TimeAbsolutePos allowed, NoteOnset disallowed (must have time first)
    CHECK(mask[vocab.encode(TokenType::TimeAbsolutePos, 0)] == false);
    CHECK(mask[vocab.encode(TokenType::NoteOnset, 0)] == true);
    // BarEnd disallowed because require_notes_ defaults to true
    CHECK(mask[vocab.encode(TokenType::BarEnd, 0)] == true);
    
    // Step with TimeAbsolutePos
    graph.step(vocab.encode(TokenType::TimeAbsolutePos, 0), vocab);
    mask = graph.get_mask(vocab);

    // Now NoteOnset allowed (since VelocityLevel is not in this vocab config)
    CHECK(mask[vocab.encode(TokenType::NoteOnset, 0)] == false);
    
    // Add Attribute constraint: Force NoteOnset to be exactly 60
    graph.add_constraint(std::make_shared<AttributeValueConstraint>(TokenType::NoteOnset, 60));
    mask = graph.get_mask(vocab);
    
    CHECK(mask[vocab.encode(TokenType::NoteOnset, 60)] == false);
    CHECK(mask[vocab.encode(TokenType::NoteOnset, 61)] == true);
    CHECK(mask[vocab.encode(TokenType::NoteOnset, 0)] == true);
}
