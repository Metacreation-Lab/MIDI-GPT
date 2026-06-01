#include "grammar_constraint.h"

namespace midigpt::masking {

void GrammarConstraint::step(int token, const tokenizer::Vocabulary& vocab) {
    auto [type, value] = vocab.decode(token);
    current_state_ = type;

    switch (type) {
        case TokenType::Track:
            track_count_++;
            is_drum_ = (value == 1);
            bar_count_ = 0;
            break;
        case TokenType::Bar:
            bar_count_++;
            in_bar_ = true;
            timestep_ = -1;
            beat_length_ = 4;
            has_notes_in_block_ = false;
            break;
        case TokenType::TimeSig:
            if (vocab.config().time_signatures) {
                auto [num, den] = vocab.config().time_signatures->decode(value);
                beat_length_ = 4 * num / den;
            }
            break;
        case TokenType::BarEnd:
            in_bar_ = false;
            break;
        case TokenType::TimeAbsolutePos:
            timestep_ = value;
            break;
        case TokenType::FillInStart:
            in_infill_ = true;
            has_notes_in_block_ = false;
            timestep_ = -1; // Reset time to start of fill block
            break;
        case TokenType::FillInEnd:
            in_infill_ = false;
            break;
        case TokenType::NoteOnset:
        case TokenType::NotePitch:
            has_notes_in_block_ = true;
            break;
        default:
            break;
    }
}

void GrammarConstraint::apply(std::vector<bool>& mask, const tokenizer::Vocabulary& vocab) const {
    auto allow = [&](TokenType type) {
        auto [start, end] = vocab.range(type);
        if (start != -1) {
            for (int i = start; i < end; ++i) mask[i] = false;
        }
    };

    // Default: everything disallowed
    std::fill(mask.begin(), mask.end(), true);

    switch (current_state_) {

        case TokenType::PieceStart:
            allow(TokenType::NumBars);
            allow(TokenType::Track);
            break;

        case TokenType::NumBars:
            allow(TokenType::Track);
            break;

        case TokenType::Track:
            allow(TokenType::Instrument);
            if (exact_bars_ < 0) {
                allow(TokenType::Bar);
                allow(TokenType::TrackEnd);
            } else if (bar_count_ < exact_bars_) {
                allow(TokenType::Bar);
            } else {
                allow(TokenType::TrackEnd);
            }
            break;

        case TokenType::Instrument:
            allow(TokenType::NoteDensity);
            allow(TokenType::MinPolyphony);
            allow(TokenType::MaxPolyphony);
            allow(TokenType::MinNoteDuration);
            allow(TokenType::MaxNoteDuration);
            allow(TokenType::OnsetPolyphony);
            allow(TokenType::PitchRange);
            if (exact_bars_ < 0) {
                allow(TokenType::Bar);
                allow(TokenType::TrackEnd);
            } else if (bar_count_ < exact_bars_) {
                allow(TokenType::Bar);
            } else {
                allow(TokenType::TrackEnd);
            }
            break;

        // After any track-level attribute, allow more attrs or bar
        case TokenType::NoteDensity:
        case TokenType::MinPolyphony:
        case TokenType::MaxPolyphony:
        case TokenType::MinNoteDuration:
        case TokenType::MaxNoteDuration:
        case TokenType::OnsetPolyphony:
        case TokenType::PitchRange:
            allow(TokenType::NoteDensity);
            allow(TokenType::MinPolyphony);
            allow(TokenType::MaxPolyphony);
            allow(TokenType::MinNoteDuration);
            allow(TokenType::MaxNoteDuration);
            allow(TokenType::OnsetPolyphony);
            allow(TokenType::PitchRange);
            if (exact_bars_ < 0) {
                allow(TokenType::Bar);
                allow(TokenType::TrackEnd);
            } else if (bar_count_ < exact_bars_) {
                allow(TokenType::Bar);
            } else {
                allow(TokenType::TrackEnd);
            }
            break;

        case TokenType::Bar:
            allow(TokenType::TimeSig);
            allow(TokenType::Tension);
            allow(TokenType::PitchClassSet);
            allow(TokenType::MaskBar);
            allow(TokenType::TimeAbsolutePos);
            // Encoder skips TimeAbsolutePos when onset==0, so allow direct
            // entry into the note tokens at bar start (position 0).
            allow(TokenType::VelocityLevel);
            allow(TokenType::NoteOnset);
            allow(TokenType::NotePitch);
            allow(TokenType::DeltaDirection);
            allow(TokenType::Delta);
            if (!require_notes_ || has_notes_in_block_) allow(TokenType::BarEnd);
            allow(TokenType::FillInPlaceholder);
            break;

        case TokenType::TimeSig:
            allow(TokenType::Tension);
            allow(TokenType::PitchClassSet);
            allow(TokenType::MaskBar);
            allow(TokenType::TimeAbsolutePos);
            allow(TokenType::VelocityLevel);
            allow(TokenType::NoteOnset);
            allow(TokenType::NotePitch);
            allow(TokenType::DeltaDirection);
            allow(TokenType::Delta);
            if (!require_notes_ || has_notes_in_block_) allow(TokenType::BarEnd);
            allow(TokenType::FillInPlaceholder);
            break;

        case TokenType::Tension:
        case TokenType::PitchClassSet:
            allow(TokenType::Tension);
            allow(TokenType::PitchClassSet);
            allow(TokenType::MaskBar);
            allow(TokenType::TimeAbsolutePos);
            allow(TokenType::VelocityLevel);
            allow(TokenType::NoteOnset);
            allow(TokenType::NotePitch);
            allow(TokenType::DeltaDirection);
            allow(TokenType::Delta);
            if (!require_notes_ || has_notes_in_block_) allow(TokenType::BarEnd);
            break;

        case TokenType::MaskBar:
            allow(TokenType::BarEnd);
            break;

        case TokenType::FillInPlaceholder:
            allow(TokenType::BarEnd);
            break;

        case TokenType::BarEnd:
            // Exact-bar enforcement for autoregressive: only allow track/piece
            // end when we have generated exactly the requested number of bars.
            allow(TokenType::FillInStart);
            if (exact_bars_ < 0) {
                allow(TokenType::TrackEnd);
                allow(TokenType::PieceEnd);
                allow(TokenType::Bar);
            } else if (bar_count_ < exact_bars_) {
                allow(TokenType::Bar);
            } else {
                allow(TokenType::TrackEnd);
                allow(TokenType::PieceEnd);
            }
            break;

        case TokenType::TimeAbsolutePos:
            // After time position: velocity may change, or carry from previous
            // VelocityLevel — orig allows TimeAbsolutePos -> NoteOnset directly.
            allow(TokenType::VelocityLevel);
            allow(TokenType::NoteOnset);
            allow(TokenType::NotePitch);
            allow(TokenType::DeltaDirection);
            allow(TokenType::Delta);
            if (in_infill_) {
                if (!require_notes_ || has_notes_in_block_) allow(TokenType::FillInEnd);
            } else {
                if (!require_notes_ || has_notes_in_block_) allow(TokenType::BarEnd);
            }
            break;

        case TokenType::VelocityLevel:
            // orig: VelocityLevel -> {NoteOnset, Delta}
            allow(TokenType::NoteOnset);
            allow(TokenType::NotePitch);
            allow(TokenType::Delta);
            break;

        case TokenType::DeltaDirection:
            allow(TokenType::Delta);
            break;

        case TokenType::Delta:
            // orig: Delta -> {Delta, DeltaDirection, NoteOnset, FillInEnd}
            allow(TokenType::Delta);
            allow(TokenType::DeltaDirection);
            allow(TokenType::NoteOnset);
            allow(TokenType::NotePitch);
            if (in_infill_) {
                if (!require_notes_ || has_notes_in_block_) allow(TokenType::FillInEnd);
            }
            break;

        case TokenType::NoteOnset:
        case TokenType::NotePitch:
            if (is_drum_) {
                // Drums: no duration token, go to next note or end
                allow(TokenType::VelocityLevel);
                allow(TokenType::NoteOnset);
                allow(TokenType::NotePitch);
                allow(TokenType::TimeAbsolutePos);
                allow(TokenType::DeltaDirection);
                allow(TokenType::Delta);
                if (in_infill_) {
                    allow(TokenType::FillInEnd);
                } else {
                    allow(TokenType::BarEnd);
                }
            } else {
                // Melodic: must have duration next
                allow(TokenType::NoteDuration);
            }
            break;

        case TokenType::NoteDuration:
            // After duration: another note (same onset), new time, or end
            allow(TokenType::VelocityLevel);
            allow(TokenType::NoteOnset);
            allow(TokenType::NotePitch);
            allow(TokenType::TimeAbsolutePos);
            allow(TokenType::DeltaDirection);
            allow(TokenType::Delta);
            if (in_infill_) {
                allow(TokenType::FillInEnd);
            } else {
                allow(TokenType::BarEnd);
            }
            break;

        case TokenType::TrackEnd:
            allow(TokenType::PieceEnd);
            if (max_tracks_ < 0 || track_count_ < max_tracks_) {
                allow(TokenType::Track);
            }
            break;

        case TokenType::FillInStart:
            // orig: FillInStart -> {TimeAbsolutePos, VelocityLevel} only.
            // Forbidding immediate FillInEnd prevents early termination with 0 notes.
            allow(TokenType::TimeAbsolutePos);
            allow(TokenType::VelocityLevel);
            break;

        case TokenType::FillInEnd:
            allow(TokenType::BarEnd);
            allow(TokenType::FillInStart);
            allow(TokenType::PieceEnd);
            break;

        default:
            // Safety fallback: allow everything
            std::fill(mask.begin(), mask.end(), false);
            break;
    }

    // TimeAbsolutePos: enforce monotonicity and bar boundary
    if (in_bar_ && vocab.has(TokenType::TimeAbsolutePos)) {
        auto [tap_start, tap_end] = vocab.range(TokenType::TimeAbsolutePos);
        if (tap_start != -1) {
            int bar_ticks = beat_length_ * vocab.config().resolution;
            for (int i = tap_start; i < tap_end; ++i) {
                auto [_, pos] = vocab.decode(i);
                if (pos <= timestep_ || pos >= bar_ticks) {
                    mask[i] = true;
                }
            }
        }
    }

    // Autoregressive mode: forbid every FillIn-* token everywhere. The
    // encoder turns off multi-fill in AR mode (no FillInPlaceholder in the
    // prompt), so the model has no business emitting FillInStart/FillInEnd.
    if (is_autoregressive_) {
        for (TokenType t : {TokenType::FillInStart, TokenType::FillInEnd,
                            TokenType::FillInPlaceholder}) {
            auto [s, e] = vocab.range(t);
            if (s != -1) for (int i = s; i < e; ++i) mask[i] = true;
        }
    }

    // Mask track start/end if requested (single-track generation)
    if (mask_track_start_) {
        auto [start, end] = vocab.range(TokenType::Track);
        if (start != -1) for (int i = start; i < end; ++i) mask[i] = true;
    }
    if (mask_track_end_) {
        auto [start, end] = vocab.range(TokenType::TrackEnd);
        if (start != -1) for (int i = start; i < end; ++i) mask[i] = true;
    }
}

} // namespace midigpt::masking
