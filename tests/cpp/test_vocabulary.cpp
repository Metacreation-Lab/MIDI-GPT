#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "../../src/cpp/tokenizer/vocabulary.h"

using namespace midigpt;
using namespace midigpt::tokenizer;

TEST_CASE("Vocabulary operations") {
    EncoderConfig config;
    config.token_domains.push_back({TokenType::PieceStart, 1});
    config.token_domains.push_back({TokenType::Track, 10});
    config.token_domains.push_back({TokenType::NoteOnset, 128});
    
    Vocabulary vocab(config);
    
    CHECK(vocab.size() == 139);
    CHECK(vocab.has(TokenType::PieceStart));
    CHECK(vocab.has(TokenType::Track));
    CHECK(vocab.has(TokenType::NoteOnset));
    CHECK(!vocab.has(TokenType::PieceEnd));
    
    CHECK(vocab.domain_size(TokenType::PieceStart) == 1);
    CHECK(vocab.domain_size(TokenType::Track) == 10);
    CHECK(vocab.domain_size(TokenType::NoteOnset) == 128);
    
    CHECK(vocab.offset(TokenType::PieceStart) == 0);
    CHECK(vocab.offset(TokenType::Track) == 1);
    CHECK(vocab.offset(TokenType::NoteOnset) == 11);
    
    CHECK(vocab.encode(TokenType::Track, 5) == 6);
    CHECK(vocab.encode(TokenType::NoteOnset, 60) == 71);
    
    auto decoded = vocab.decode(71);
    CHECK(decoded.first == TokenType::NoteOnset);
    CHECK(decoded.second == 60);
    
    CHECK(vocab.is_type(71, TokenType::NoteOnset));
    CHECK(!vocab.is_type(71, TokenType::Track));
    
    CHECK(vocab.get_type(6) == TokenType::Track);
    
    auto r = vocab.range(TokenType::NoteOnset);
    CHECK(r.first == 11);
    CHECK(r.second == 139);
}

TEST_CASE("EncoderConfig JSON serialization") {
    EncoderConfig config;
    config.resolution = 960;
    config.emit_delta_tokens = true;
    config.token_domains.push_back({TokenType::PieceStart, 1});
    config.token_domains.push_back({TokenType::Track, 10});
    
    std::string json_str = config.to_json();
    EncoderConfig config2 = EncoderConfig::from_json(json_str);
    
    CHECK(config2.resolution == 960);
    CHECK(config2.emit_delta_tokens == true);
    CHECK(config2.token_domains.size() == 2);
    CHECK(config2.token_domains[0].type == TokenType::PieceStart);
    CHECK(config2.token_domains[0].domain_size == 1);
    CHECK(config2.token_domains[1].type == TokenType::Track);
    CHECK(config2.token_domains[1].domain_size == 10);
}
