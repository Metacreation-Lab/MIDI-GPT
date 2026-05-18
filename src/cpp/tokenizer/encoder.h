#pragma once

#include "vocabulary.h"
#include "../core/score.h"
#include <set>
#include <utility>
#include <vector>

namespace midigpt::tokenizer {

// Per-call encoding options. These describe the request, not the encoder.
// They never live in EncoderConfig (which is fixed by the checkpoint).
struct EncodeOptions {
    // Suffix-autoregressive: encode only a prefix of this track's bars,
    // omitting TRACK_END so the model can continue generating.
    int partial_encode_track_index = -1;
    int partial_encode_track_bars  = -1;

    // Multi-fill / bar infill: set of (track_index, bar_index) pairs to infill.
    // Bars in this set get FILL_IN_PLACEHOLDER during normal encoding,
    // then their notes are appended as FILL_IN_START ... FILL_IN_END blocks.
    // Non-empty requires EncoderConfig::supports_infill == true.
    std::set<std::pair<int,int>> multi_fill;

    // Window size (bars) for this encode call — emitted as the NumBars token.
    // 0 means "not set"; the encoder falls back to EncoderConfig::model_dim.
    // Validation that this matches num_bars_map happens at the request layer.
    int window_bars = 0;
};

class Encoder {
public:
    explicit Encoder(const Vocabulary& vocab);

    // Encodes a full score into a linear sequence of tokens.
    std::vector<int> encode(const Score& score,
                            const EncodeOptions& opts = {}) const;

private:
    const Vocabulary& vocab_;
};

} // namespace midigpt::tokenizer
