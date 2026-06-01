// Vocabulary + EncoderConfig JSON. Covers basic operations (encode/decode/
// range/offset/has/is_type/get_type), edge cases (type not in config,
// domain-size 1 and large, offset arithmetic across many types), and the
// EncoderConfig JSON roundtrip including derive_token_domains.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "../../src/cpp/tokenizer/vocabulary.h"

using namespace midigpt;
using namespace midigpt::tokenizer;
using TT = midigpt::TokenType;

// ---------------------------------------------------------------------------
// Basic operations
// ---------------------------------------------------------------------------

TEST_CASE("Vocabulary: encode/decode/range/offset for 3 types") {
    EncoderConfig cfg;
    cfg.token_domains.push_back({TT::PieceStart, 1});
    cfg.token_domains.push_back({TT::Track, 10});
    cfg.token_domains.push_back({TT::NoteOnset, 128});
    Vocabulary vocab(cfg);

    CHECK(vocab.size() == 139);
    CHECK(vocab.has(TT::PieceStart));
    CHECK(vocab.has(TT::Track));
    CHECK(vocab.has(TT::NoteOnset));
    CHECK(!vocab.has(TT::PieceEnd));

    CHECK(vocab.domain_size(TT::PieceStart) == 1);
    CHECK(vocab.domain_size(TT::Track) == 10);
    CHECK(vocab.domain_size(TT::NoteOnset) == 128);

    CHECK(vocab.offset(TT::PieceStart) == 0);
    CHECK(vocab.offset(TT::Track) == 1);
    CHECK(vocab.offset(TT::NoteOnset) == 11);

    CHECK(vocab.encode(TT::Track, 5) == 6);
    CHECK(vocab.encode(TT::NoteOnset, 60) == 71);

    auto d = vocab.decode(71);
    CHECK(d.first == TT::NoteOnset);
    CHECK(d.second == 60);

    CHECK(vocab.is_type(71, TT::NoteOnset));
    CHECK_FALSE(vocab.is_type(71, TT::Track));

    CHECK(vocab.get_type(6) == TT::Track);

    auto r = vocab.range(TT::NoteOnset);
    CHECK(r.first == 11);
    CHECK(r.second == 139);
}

// ---------------------------------------------------------------------------
// Type not in config
// ---------------------------------------------------------------------------

TEST_CASE("Vocabulary: has() returns false for absent type") {
    EncoderConfig cfg;
    cfg.token_domains.push_back({TT::Track, 5});
    Vocabulary vocab(cfg);
    CHECK_FALSE(vocab.has(TT::Tension));
    CHECK_FALSE(vocab.has(TT::PieceEnd));
    CHECK_FALSE(vocab.has(TT::NoteOnset));
}

TEST_CASE("Vocabulary: range() for absent type returns (-1,-1)") {
    EncoderConfig cfg;
    cfg.token_domains.push_back({TT::Track, 5});
    Vocabulary vocab(cfg);
    auto r = vocab.range(TT::Tension);
    CHECK(r.first == -1);
    CHECK(r.second == -1);
}

// ---------------------------------------------------------------------------
// Offset / size arithmetic across many types
// ---------------------------------------------------------------------------

TEST_CASE("Vocabulary: offset(t_{i+1}) == offset(t_i) + domain_size(t_i)") {
    EncoderConfig cfg;
    std::vector<std::pair<TT,int>> ds = {
        {TT::PieceStart, 1},
        {TT::Track,      10},
        {TT::Instrument, 128},
        {TT::Bar,        1},
        {TT::TimeAbsolutePos, 192},
        {TT::VelocityLevel,  32},
        {TT::NoteOnset,  128},
        {TT::NoteDuration, 64},
        {TT::BarEnd,     1},
        {TT::TrackEnd,   1},
        {TT::PieceEnd,   1},
    };
    for (auto& d : ds) cfg.token_domains.push_back({d.first, d.second});
    Vocabulary vocab(cfg);

    int expected = 0;
    for (auto& d : ds) {
        CHECK(vocab.offset(d.first) == expected);
        CHECK(vocab.domain_size(d.first) == d.second);
        auto r = vocab.range(d.first);
        CHECK(r.first == expected);
        CHECK(r.second == expected + d.second);
        expected += d.second;
    }
    CHECK(vocab.size() == expected);
}

// ---------------------------------------------------------------------------
// Domain-size edges: 1 and large
// ---------------------------------------------------------------------------

TEST_CASE("Vocabulary: domain size 1 yields single-token range") {
    EncoderConfig cfg;
    cfg.token_domains.push_back({TT::PieceStart, 1});
    Vocabulary vocab(cfg);
    CHECK(vocab.size() == 1);
    CHECK(vocab.encode(TT::PieceStart, 0) == 0);
    auto r = vocab.range(TT::PieceStart);
    CHECK(r.second - r.first == 1);
}

TEST_CASE("Vocabulary: large domain (4096) encodes/decodes correctly") {
    EncoderConfig cfg;
    cfg.token_domains.push_back({TT::PieceStart, 1});
    cfg.token_domains.push_back({TT::PitchClassSet, 4096});
    Vocabulary vocab(cfg);
    for (int v : {0, 1, 1000, 4095}) {
        int tok = vocab.encode(TT::PitchClassSet, v);
        auto d = vocab.decode(tok);
        CHECK(d.first == TT::PitchClassSet);
        CHECK(d.second == v);
    }
}

// ---------------------------------------------------------------------------
// is_type / get_type at boundaries
// ---------------------------------------------------------------------------

TEST_CASE("Vocabulary: get_type at every offset and is_type at boundaries") {
    EncoderConfig cfg;
    cfg.token_domains.push_back({TT::Track, 4});
    cfg.token_domains.push_back({TT::Bar, 2});
    cfg.token_domains.push_back({TT::NoteOnset, 3});
    Vocabulary vocab(cfg);
    // First token of each range
    CHECK(vocab.get_type(vocab.encode(TT::Track, 0)) == TT::Track);
    CHECK(vocab.get_type(vocab.encode(TT::Track, 3)) == TT::Track);
    CHECK(vocab.get_type(vocab.encode(TT::Bar, 0)) == TT::Bar);
    CHECK(vocab.get_type(vocab.encode(TT::Bar, 1)) == TT::Bar);
    CHECK(vocab.get_type(vocab.encode(TT::NoteOnset, 0)) == TT::NoteOnset);
    CHECK(vocab.get_type(vocab.encode(TT::NoteOnset, 2)) == TT::NoteOnset);
    // Cross-type
    CHECK_FALSE(vocab.is_type(vocab.encode(TT::Track, 3), TT::Bar));
    CHECK_FALSE(vocab.is_type(vocab.encode(TT::Bar, 0), TT::Track));
}

// ---------------------------------------------------------------------------
// EncoderConfig JSON roundtrip
// ---------------------------------------------------------------------------

TEST_CASE("EncoderConfig JSON: roundtrip preserves primitives and domains") {
    EncoderConfig cfg;
    cfg.resolution = 960;
    cfg.emit_delta_tokens = true;
    cfg.velocity_levels = 64;
    cfg.token_domains.push_back({TT::PieceStart, 1});
    cfg.token_domains.push_back({TT::Track, 10});

    auto json_str = cfg.to_json();
    auto cfg2 = EncoderConfig::from_json(json_str);

    CHECK(cfg2.resolution == 960);
    CHECK(cfg2.emit_delta_tokens == true);
    CHECK(cfg2.velocity_levels == 64);
    REQUIRE(cfg2.token_domains.size() == 2);
    CHECK(cfg2.token_domains[0].type == TT::PieceStart);
    CHECK(cfg2.token_domains[0].domain_size == 1);
    CHECK(cfg2.token_domains[1].type == TT::Track);
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
        if (d.type == TT::Tension && d.domain_size == 10)    found_tension = true;
        if (d.type == TT::NoteDensity && d.domain_size == 8) found_density = true;
    }
    CHECK(found_tension);
    CHECK(found_density);
}
