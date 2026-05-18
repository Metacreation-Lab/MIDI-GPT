#pragma once

#include "constraint.h"

namespace midigpt::masking {

class GrammarConstraint : public Constraint {
public:
    GrammarConstraint() = default;

    void step(int token, const tokenizer::Vocabulary& vocab) override;
    void apply(std::vector<bool>& mask, const tokenizer::Vocabulary& vocab) const override;
    void set_mask_track_start(bool mask) { mask_track_start_ = mask; }
    void set_mask_track_end(bool mask) { mask_track_end_ = mask; }
    // Exact-bar enforcement for autoregressive: when >= 0, the grammar will
    // mask TrackEnd/PieceEnd until bar_count_ reaches this value, and mask
    // additional Bar tokens once it does — forcing exactly N bars per track.
    void set_exact_bars(int exact_bars) { exact_bars_ = exact_bars; }
    // Backwards-compatible alias (old name).
    void set_max_bars(int max_bars) { exact_bars_ = max_bars; }
    void set_max_tracks(int max_tracks) { max_tracks_ = max_tracks; }
    void set_require_notes(bool require) { require_notes_ = require; }
    void set_fillin_drum(bool is_drum) override { is_drum_ = is_drum; }
    // Autoregressive mode: forbid all FillIn-* tokens (multi-fill is off, model
    // must terminate via TrackEnd/PieceEnd once exact_bars_ is reached).
    void set_autoregressive_mode(bool is_ar) { is_autoregressive_ = is_ar; }

private:
    TokenType current_state_ = TokenType::PieceStart;
    bool mask_track_start_ = false;
    bool mask_track_end_ = false;

    int track_count_ = 0;
    int bar_count_ = 0;
    int max_tracks_ = -1;
    int exact_bars_ = -1;   // -1 = no constraint, else require exactly N Bars per track
    int timestep_ = -1;
    int beat_length_ = 4;
    bool is_drum_ = false;
    bool in_bar_ = false;
    bool in_infill_ = false;
    bool has_notes_in_block_ = false;
    bool require_notes_ = true;
    bool is_autoregressive_ = false;
};

} // namespace midigpt::masking
