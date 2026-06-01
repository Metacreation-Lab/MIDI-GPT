#pragma once

#include "constraint.h"

namespace midigpt::masking {

// Per-bar variant of AttributeValueConstraint. Tracks its own (track, bar)
// cursor by observing Track/Bar tokens, and only applies the value mask when
// the cursor matches the configured (track_ordinal, bar_index) pair.
//
// Coordinate semantics:
//   - track_ordinal: 0-based position of the target track within the current
//     step's emit sequence (i.e. count of Track tokens seen so far, minus 1).
//     For tracks_per_step=1 this is always 0.
//   - bar_index:     0-based relative bar index within the target track in
//     this step (i.e. count of Bar tokens seen since the last Track token,
//     minus 1). For full-AR-on-the-whole-track this equals the absolute bar
//     index; for partial-AR/infill the caller is responsible for converting
//     absolute -> relative.
//
// The constraint is a no-op until the cursor lands on (track_ordinal,
// bar_index); it then masks every token of `target_type_` except the one
// corresponding to `allowed_value_`.
class BarAttributeValueConstraint : public Constraint {
public:
    BarAttributeValueConstraint(TokenType target_type, int track_ordinal,
                                int bar_index, int allowed_value)
        : target_type_(target_type),
          target_track_ordinal_(track_ordinal),
          target_bar_index_(bar_index),
          allowed_value_(allowed_value) {}

    void step(int token, const tokenizer::Vocabulary& vocab) override {
        auto [type, value] = vocab.decode(token);
        if (type == TokenType::Track) {
            track_count_++;
            bar_count_ = 0;
        } else if (type == TokenType::Bar) {
            bar_count_++;
        }
    }

    void apply(std::vector<bool>& mask, const tokenizer::Vocabulary& vocab) const override {
        // Cursor: 0-based ordinal of the track/bar we are currently inside.
        // After the Nth Track token we are inside track ordinal (N-1); same
        // for bars within the current track.
        int cur_track = track_count_ - 1;
        int cur_bar   = bar_count_ - 1;
        if (cur_track != target_track_ordinal_ || cur_bar != target_bar_index_) {
            return;
        }
        auto [start, end] = vocab.range(target_type_);
        if (start == -1) return;
        for (int i = start; i < end; ++i) {
            if (i - start != allowed_value_) {
                mask[i] = true;
            }
        }
    }

private:
    TokenType target_type_;
    int target_track_ordinal_;
    int target_bar_index_;
    int allowed_value_;
    int track_count_ = 0;
    int bar_count_   = 0;
};

} // namespace midigpt::masking
