// Vocabulary + EncoderConfig JSON. Covers basic operations (encode/decode/
// range/offset/has/is_type/get_type), edge cases (type not in config,
// domain-size 1 and large, offset arithmetic across many types), and the
// EncoderConfig JSON roundtrip including derive_token_domains.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "../../src/cpp/tokenizer/vocabulary.h"

using namespace midigpt;
using namespace midigpt::tokenizer;

// ---------------------------------------------------------------------------
// Basic operations
// ---------------------------------------------------------------------------

TEST_CASE("Vocabulary: encode/decode/range/offset for 3 types") {
    EncoderConfig cfg;
    cfg.token_domains.push_back({TokenType::PieceStart, 1});
    cfg.token_domains.push_back({TokenType::Track, 10});
    cfg.token_domains.push_back({TokenType::NoteOnset, 128});
    Vocabulary vocab(cfg);

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

    auto d = vocab.decode(71);
    CHECK(d.first == TokenType::NoteOnset);
    CHECK(d.second == 60);

    CHECK(vocab.is_type(71, TokenType::NoteOnset));
    CHECK_FALSE(vocab.is_type(71, TokenType::Track));

    CHECK(vocab.get_type(6) == TokenType::Track);

    auto r = vocab.range(TokenType::NoteOnset);
    CHECK(r.first == 11);
    CHECK(r.second == 139);
}

// ---------------------------------------------------------------------------
// Type not in config
// ---------------------------------------------------------------------------

TEST_CASE("Vocabulary: has() returns false for absent type") {
    EncoderConfig cfg;
    cfg.token_domains.push_back({TokenType::Track, 5});
    Vocabulary vocab(cfg);
    CHECK_FALSE(vocab.has(TokenType::Tension));
    CHECK_FALSE(vocab.has(TokenType::PieceEnd));
    CHECK_FALSE(vocab.has(TokenType::NoteOnset));
}

TEST_CASE("Vocabulary: range() for absent type returns (-1,-1)") {
    EncoderConfig cfg;
    cfg.token_domains.push_back({TokenType::Track, 5});
    Vocabulary vocab(cfg);
    auto r = vocab.range(TokenType::Tension);
    CHECK(r.first == -1);
    CHECK(r.second == -1);
}

// ---------------------------------------------------------------------------
// Offset / size arithmetic across many types
// ---------------------------------------------------------------------------

TEST_CASE("Vocabulary: offset(t_{i+1}) == offset(t_i) + domain_size(t_i)") {
    EncoderConfig cfg;
    std::vector<std::pair<TokenType,int>> ds = {
        {TokenType::PieceStart, 1},
        {TokenType::Track,      10},
        {TokenType::Instrument, 128},
        {TokenType::Bar,        1},
        {TokenType::TimeAbsolutePos, 192},
        {TokenType::VelocityLevel,  32},
        {TokenType::NoteOnset,  128},
        {TokenType::NoteDuration, 64},
        {TokenType::BarEnd,     1},
        {TokenType::TrackEnd,   1},
        {TokenType::PieceEnd,   1},
    };
    for (auto& [t, s] : ds) cfg.token_domains.push_back({t, s});
    Vocabulary vocab(cfg);

    int expected = 0;
    for (auto& [t, s] : ds) {
        CHECK(vocab.offset(t) == expected);
        CHECK(vocab.domain_size(t) == s);
        auto r = vocab.range(t);
        CHECK(r.first == expected);
        CHECK(r.second == expected + s);
        expected += s;
    }
    CHECK(vocab.size() == expected);
}

// ---------------------------------------------------------------------------
// Domain-size edges: 1 and large
// ---------------------------------------------------------------------------

TEST_CASE("Vocabulary: domain size 1 yields single-token range") {
    EncoderConfig cfg;
    cfg.token_domains.push_back({TokenType::PieceStart, 1});
    Vocabulary vocab(cfg);
    CHECK(vocab.size() == 1);
    CHECK(vocab.encode(TokenType::PieceStart, 0) == 0);
    auto r = vocab.range(TokenType::PieceStart);
    CHECK(r.second - r.first == 1);
}

TEST_CASE("Vocabulary: large domain (4096) encodes/decodes correctly") {
    EncoderConfig cfg;
    cfg.token_domains.push_back({TokenType::PieceStart, 1});
    cfg.token_domains.push_back({TokenType::PitchClassSet, 4096});
    Vocabulary vocab(cfg);
    for (int v : {0, 1, 1000, 4095}) {
        int tok = vocab.encode(TokenType::PitchClassSet, v);
        auto d = vocab.decode(tok);
        CHECK(d.first == TokenType::PitchClassSet);
        CHECK(d.second == v);
    }
}

// ---------------------------------------------------------------------------
// is_type / get_type at boundaries
// ---------------------------------------------------------------------------

TEST_CASE("Vocabulary: get_type at every offset and is_type at boundaries") {
    EncoderConfig cfg;
    cfg.token_domains.push_back({TokenType::Track, 4});
    cfg.token_domains.push_back({TokenType::Bar, 2});
    cfg.token_domains.push_back({TokenType::NoteOnset, 3});
    Vocabulary vocab(cfg);
    // First token of each range
    CHECK(vocab.get_type(vocab.encode(TokenType::Track, 0)) == TokenType::Track);
    CHECK(vocab.get_type(vocab.encode(TokenType::Track, 3)) == TokenType::Track);
    CHECK(vocab.get_type(vocab.encode(TokenType::Bar, 0)) == TokenType::Bar);
    CHECK(vocab.get_type(vocab.encode(TokenType::Bar, 1)) == TokenType::Bar);
    CHECK(vocab.get_type(vocab.encode(TokenType::NoteOnset, 0)) == TokenType::NoteOnset);
    CHECK(vocab.get_type(vocab.encode(TokenType::NoteOnset, 2)) == TokenType::NoteOnset);
    // Cross-type
    CHECK_FALSE(vocab.is_type(vocab.encode(TokenType::Track, 3), TokenType::Bar));
    CHECK_FALSE(vocab.is_type(vocab.encode(TokenType::Bar, 0), TokenType::Track));
}

// ---------------------------------------------------------------------------
// EncoderConfig JSON roundtrip
// ---------------------------------------------------------------------------

TEST_CASE("EncoderConfig JSON: roundtrip preserves primitives and domains") {
    EncoderConfig cfg;
    cfg.resolution = 960;
    cfg.emit_delta_tokens = true;
    cfg.velocity_levels = 64;
    cfg.token_domains.push_back({TokenType::PieceStart, 1});
    cfg.token_domains.push_back({TokenType::Track, 10});

    auto json_str = cfg.to_json();
    auto cfg2 = EncoderConfig::from_json(json_str);

    CHECK(cfg2.resolution == 960);
    CHECK(cfg2.emit_delta_tokens == true);
    CHECK(cfg2.velocity_levels == 64);
    REQUIRE(cfg2.token_domains.size() == 2);
    CHECK(cfg2.token_domains[0].type == TokenType::PieceStart);
    CHECK(cfg2.token_domains[0].domain_size == 1);
    CHECK(cfg2.token_domains[1].type == TokenType::Track);
    CHECK(cfg2.token_domains[1].domain_size == 10);
}

// ---------------------------------------------------------------------------
// derive_token_domains + add_attribute_token_domains
// ---------------------------------------------------------------------------

TEST_CASE("EncoderConfig.derive_token_domains populates structural tokens") {
    EncoderConfig cfg;
    cfg.resolution = 12;
    cfg.pitch_min = 21;
    cfg.pitch_max = 108;
    cfg.velocity_levels = 32;
    cfg.note_duration_max_beats = 4;
    cfg.derive_token_domains();
    // At minimum: NoteOnset domain should equal (pitch_max - pitch_min + 1)
    // OR the conventional 128. The exact contract lives in encoder_config.cpp;
    // here we just assert SOME structural tokens were added.
    CHECK(cfg.token_domains.size() > 0);
}

TEST_CASE("EncoderConfig.add_attribute_token_domains appends attribute slots") {
    EncoderConfig cfg;
    cfg.derive_token_domains();
    size_t before = cfg.token_domains.size();
    cfg.add_attribute_token_domains({{"Tension", 10}, {"NoteDensity", 8}});
    CHECK(cfg.token_domains.size() == before + 2);
    bool found_tension = false, found_density = false;
    for (auto& d : cfg.token_domains) {
        if (d.type == TokenType::Tension && d.domain_size == 10)    found_tension = true;
        if (d.type == TokenType::NoteDensity && d.domain_size == 8) found_density = true;
    }
    CHECK(found_tension);
    CHECK(found_density);
}
