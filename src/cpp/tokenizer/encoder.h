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

    // If true, "future" bars are emitted as empty shells (Bar TimeSig BarEnd)
    // instead of a MaskBar token, and the encoder records the [start,end)
    // token range of each such shell as a "hidden span". At inference time,
    // queries are masked out of those key positions so the model cannot attend
    // to the empty shell — used to run yellow-style checkpoints (no MaskBar
    // token in vocab) on the realtime lookahead path.
    bool use_span_masks = false;

    // If true, future bars are omitted entirely from the token sequence
    // (no tokens emitted at all). Used by the "remove" and "attention_skip"
    // mask modes where the caller builds a filtered token sequence directly.
    bool remove_future_bars = false;
};

struct EncodeResult {
    std::vector<int>                  tokens;
    std::vector<std::pair<int,int>>   hidden_spans;  // [start, end) per masked bar
};

class Encoder {
public:
    explicit Encoder(const Vocabulary& vocab);

    // Encodes a full score into a linear sequence of tokens. Convenience
    // wrapper around encode_full() that discards span metadata.
    std::vector<int> encode(const Score& score,
                            const EncodeOptions& opts = {}) const;

    // Encodes and also returns per-bar hidden spans when use_span_masks=true.
    EncodeResult encode_full(const Score& score,
                             const EncodeOptions& opts = {}) const;

private:
    const Vocabulary& vocab_;
};

} // namespace midigpt::tokenizer
