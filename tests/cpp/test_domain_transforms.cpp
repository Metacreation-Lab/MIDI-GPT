#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "../../src/cpp/tokenizer/domain_transforms.h"

using namespace midigpt::tokenizer;

// ============================================================
// VelocityQuantizer
// ============================================================

TEST_CASE("VelocityQuantizer with 32 levels matches legacy map") {
    VelocityQuantizer vq(32);

    // Legacy DEFAULT_VELOCITY_MAP (all 128 entries)
    const int LEGACY[128] = {
        0, 1,1,1,1, 2,2,2,2, 3,3,3,3, 4,4,4,4, 5,5,5,5, 6,6,6,6, 7,7,7,7,
        8,8,8,8,8, 9,9,9,9, 10,10,10,10, 11,11,11,11, 12,12,12,12, 13,13,13,13,
        14,14,14,14, 15,15,15,15, 16,16,16,16,16, 17,17,17,17, 18,18,18,18,
        19,19,19,19, 20,20,20,20, 21,21,21,21, 22,22,22,22, 23,23,23,23,
        24,24,24,24,24, 25,25,25,25, 26,26,26,26, 27,27,27,27,
        28,28,28,28, 29,29,29,29, 30,30,30,30, 31,31,31,31
    };

    for (int v = 0; v < 128; ++v) {
        CHECK(vq.encode(v) == LEGACY[v]);
    }
}

TEST_CASE("VelocityQuantizer reverse matches legacy reverse map") {
    VelocityQuantizer vq(32);

    const int LEGACY_REV[32] = {
        0, 2, 6, 10, 14, 18, 22, 26, 30, 35,
        39, 43, 47, 51, 55, 59, 63, 68, 72,
        76, 80, 84, 88, 92, 96, 101, 105, 109,
        113, 117, 121, 125
    };

    for (int level = 0; level < 32; ++level) {
        CHECK(vq.decode(level) == LEGACY_REV[level]);
    }
}

TEST_CASE("VelocityQuantizer roundtrip") {
    VelocityQuantizer vq(32);
    for (int v = 0; v < 128; ++v) {
        int level = vq.encode(v);
        int reconstructed = vq.decode(level);
        // Reconstructed should be within the same bin
        CHECK(vq.encode(reconstructed) == level);
    }
}

TEST_CASE("VelocityQuantizer works with different level counts") {
    // Should not throw
    VelocityQuantizer vq16(16);
    CHECK(vq16.num_levels() == 16);
    CHECK(vq16.encode(0) == 0);
    CHECK(vq16.encode(127) == 15);

    VelocityQuantizer vq64(64);
    CHECK(vq64.num_levels() == 64);
    CHECK(vq64.encode(0) == 0);
    CHECK(vq64.encode(127) == 63);

    // Edge: 128 levels (1:1 except vel 0)
    VelocityQuantizer vq128(128);
    CHECK(vq128.encode(0) == 0);
    CHECK(vq128.encode(127) == 127);
}

// ============================================================
// TimeSignatureList
// ============================================================

TEST_CASE("TimeSignatureList encode/decode roundtrip") {
    TimeSignatureList tsl({
        {4,4}, {3,4}, {2,4}, {6,8}, {2,2}, {1,4}
    });

    CHECK(tsl.size() == 6);
    CHECK(tsl.encode(4, 4) == 0);
    CHECK(tsl.encode(3, 4) == 1);
    CHECK(tsl.encode(2, 4) == 2);
    CHECK(tsl.encode(6, 8) == 3);
    CHECK(tsl.encode(2, 2) == 4);
    CHECK(tsl.encode(1, 4) == 5);

    auto [n0, d0] = tsl.decode(0);
    CHECK(n0 == 4);
    CHECK(d0 == 4);

    auto [n3, d3] = tsl.decode(3);
    CHECK(n3 == 6);
    CHECK(d3 == 8);
}

TEST_CASE("TimeSignatureList throws on unknown signature") {
    TimeSignatureList tsl({{4,4}, {3,4}});
    CHECK_THROWS(tsl.encode(7, 8));
}

TEST_CASE("TimeSignatureList JSON roundtrip") {
    nlohmann::json j = nlohmann::json::array({"4/4", "3/4", "6/8"});
    auto tsl = TimeSignatureList::from_json(j);

    CHECK(tsl.size() == 3);
    CHECK(tsl.encode(4, 4) == 0);
    CHECK(tsl.encode(6, 8) == 2);

    auto j2 = tsl.to_json();
    CHECK(j2.size() == 3);
    CHECK(j2[0] == "4/4");
    CHECK(j2[2] == "6/8");
}

// ============================================================
// InstrumentGrouping
// ============================================================

TEST_CASE("InstrumentGrouping matches legacy PRETRAIN_GROUPING") {
    std::vector<std::vector<int>> merge_groups = {
        {0,1,2}, {4,5}, {16,17,18}, {19,20}, {33,34},
        {36,37}, {38,39}, {48,49}, {50,51}, {62,63},
        {88,89,90,91,92,93,94,95}
    };
    InstrumentGrouping ig(merge_groups);

    CHECK(ig.num_groups() == 109);

    // Pianos 0,1,2 all map to group 0
    CHECK(ig.encode(0) == 0);
    CHECK(ig.encode(1) == 0);
    CHECK(ig.encode(2) == 0);

    // Honky-tonk is its own group
    CHECK(ig.encode(3) == 1);

    // E.Piano 1,2 share a group
    CHECK(ig.encode(4) == 2);
    CHECK(ig.encode(5) == 2);

    // All pads share one group
    CHECK(ig.encode(88) == ig.encode(95));

    // Instruments 96-127 are 1:1 after the pad group
    CHECK(ig.encode(96) == 77);
    CHECK(ig.encode(127) == 108);
}

TEST_CASE("InstrumentGrouping reverse returns representative") {
    std::vector<std::vector<int>> merge_groups = {
        {0,1,2}, {4,5}
    };
    InstrumentGrouping ig(merge_groups);

    // Group 0 = instruments [0,1,2], representative = 0
    CHECK(ig.decode(0) == 0);
    // Group 1 = instrument [3] (not merged, 1:1)
    CHECK(ig.decode(1) == 3);
    // Group 2 = instruments [4,5], representative = 4
    CHECK(ig.decode(2) == 4);
}

TEST_CASE("InstrumentGrouping JSON roundtrip") {
    nlohmann::json j = nlohmann::json::array({
        nlohmann::json::array({0, 1, 2}),
        nlohmann::json::array({16, 17, 18})
    });
    auto ig = InstrumentGrouping::from_json(j);

    CHECK(ig.encode(0) == ig.encode(1));
    CHECK(ig.encode(0) == ig.encode(2));
    CHECK(ig.encode(16) == ig.encode(17));
    CHECK(ig.encode(16) == ig.encode(18));
    CHECK(ig.encode(0) != ig.encode(16));
}
