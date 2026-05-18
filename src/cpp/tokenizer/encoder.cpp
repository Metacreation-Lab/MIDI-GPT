#include "encoder.h"
#include "domain_transforms.h"
#include <algorithm>
#include <map>
#include <vector>
#include <cmath>
#include <set>
#include <functional>

namespace midigpt::tokenizer {

Encoder::Encoder(const Vocabulary& vocab) : vocab_(vocab) {}

static void encode_bar_notes(
    std::vector<int>& tokens, const Score& score, const Bar& bar,
    bool is_drum, const EncoderConfig& config,
    const std::function<int(TokenType, int)>& clamp_encode,
    const Vocabulary& vocab_)
{
    std::map<int, std::vector<int>> notes_by_onset;
    std::vector<int> onset_order;
    for (int note_idx : bar.note_indices) {
        const auto& note = score.notes[note_idx];
        int onset = note.onset_ticks;
        if (notes_by_onset.find(onset) == notes_by_onset.end()) {
            onset_order.push_back(onset);
        }
        notes_by_onset[onset].push_back(note_idx);
    }
    // Match orig encoder: emit chord notes in ascending pitch order.
    for (auto& [_, bucket] : notes_by_onset) {
        std::sort(bucket.begin(), bucket.end(),
                  [&](int a, int b) { return score.notes[a].pitch < score.notes[b].pitch; });
    }

    int last_velocity = -1;
    for (int onset : onset_order) {
        if (onset > 0 && vocab_.has(TokenType::TimeAbsolutePos)) {
            tokens.push_back(clamp_encode(TokenType::TimeAbsolutePos, onset));
        }
        for (int ni : notes_by_onset[onset]) {
            const auto& note = score.notes[ni];
            if (vocab_.has(TokenType::VelocityLevel)) {
                int vel_domain = vocab_.domain_size(TokenType::VelocityLevel);
                VelocityQuantizer vq(vel_domain);
                int mapped_vel = vq.encode(note.velocity);
                bool emit_velocity = mapped_vel > 0
                    && (!config.velocity_sticky || mapped_vel != last_velocity);
                if (emit_velocity) {
                    tokens.push_back(clamp_encode(TokenType::VelocityLevel, mapped_vel));
                    last_velocity = mapped_vel;
                }
            }
            if (config.emit_delta_tokens && note.delta != 0) {
                int d = note.delta;
                if (d < 0 && vocab_.has(TokenType::DeltaDirection)) {
                    tokens.push_back(vocab_.encode(TokenType::DeltaDirection, 0));
                    d = -d;
                }
                if (d > 0 && vocab_.has(TokenType::Delta)) {
                    int max_delta = vocab_.domain_size(TokenType::Delta) - 1;
                    tokens.push_back(clamp_encode(TokenType::Delta, std::min(d, max_delta)));
                }
            }
            tokens.push_back(clamp_encode(TokenType::NoteOnset, note.pitch));
            if (!is_drum && vocab_.has(TokenType::NoteDuration)) {
                int dur_domain = vocab_.domain_size(TokenType::NoteDuration);
                int dur = std::min(note.duration_ticks, dur_domain) - 1;
                tokens.push_back(clamp_encode(TokenType::NoteDuration, std::max(0, dur)));
            }
        }
    }
}

std::vector<int> Encoder::encode(const Score& score,
                                 const EncodeOptions& opts) const {
    std::vector<int> tokens;
    const auto& config = vocab_.config();

    const bool do_multi_fill = !opts.multi_fill.empty();
    if (do_multi_fill && !config.supports_infill) {
        throw std::invalid_argument(
            "Encoder: multi_fill requested but config.supports_infill is false");
    }

    auto clamp_encode = [&](TokenType type, int val) {
        int domain = vocab_.domain_size(type);
        if (domain <= 0) return vocab_.encode(type, val);
        return vocab_.encode(type, std::clamp(val, 0, domain - 1));
    };

    // --- PIECE_START ---
    if (vocab_.has(TokenType::PieceStart)) {
        int ps_val = do_multi_fill
            ? std::min(1, vocab_.domain_size(TokenType::PieceStart) - 1)
            : 0;
        tokens.push_back(vocab_.encode(TokenType::PieceStart, ps_val));
    }

    // --- NUM_BARS ---
    // Window size for this encode call. Source order:
    //   1. opts.window_bars (set by the step planner / session)
    //   2. config.model_dim  (fallback for standalone encoder calls)
    // Domain validation (in num_bars_map) lives at the request layer.
    if (vocab_.has(TokenType::NumBars)) {
        int actual_bars = 0;
        for (const auto& t : score.tracks) {
            actual_bars = std::max(actual_bars, (int)t.bars.size());
        }
        int val = opts.window_bars > 0 ? opts.window_bars
                : (actual_bars > 0 ? actual_bars : config.model_dim);
        if (config.num_bars_map) {
            if (!config.num_bars_map->contains(val)) {
                std::string allowed;
                const auto& vs = config.num_bars_map->values();
                for (size_t i = 0; i < vs.size(); ++i) {
                    if (i) allowed += ", ";
                    allowed += std::to_string(vs[i]);
                }
                throw std::invalid_argument(
                    "Encoder: window_bars=" + std::to_string(val) +
                    " not in num_bars_map [" + allowed + "]");
            }
            val = config.num_bars_map->encode(val);
        }
        tokens.push_back(clamp_encode(TokenType::NumBars, val));
    }

    // --- TRACKS ---
    for (size_t track_idx = 0; track_idx < score.tracks.size(); ++track_idx) {
        const auto& track = score.tracks[track_idx];
        bool is_drum = (track.type == TrackType::Drum);

        // TRACK token: hardcoded 0 = melodic, 1 = drum.
        if (vocab_.has(TokenType::Track)) {
            int val = is_drum ? 1 : 0;
            tokens.push_back(clamp_encode(TokenType::Track, val));
        }

        // INSTRUMENT token
        if (vocab_.has(TokenType::Instrument)) {
            int inst = config.instrument_grouping
                ? config.instrument_grouping->encode(track.instrument)
                : track.instrument;
            tokens.push_back(clamp_encode(TokenType::Instrument, inst));
        }

        // Track-level attribute tokens AFTER instrument (matches original order)
        const std::vector<std::pair<std::string, TokenType>> post_inst_attrs = {
            {"min_polyphony",     TokenType::MinPolyphony},
            {"max_polyphony",     TokenType::MaxPolyphony},
            {"min_note_duration", TokenType::MinNoteDuration},
            {"max_note_duration", TokenType::MaxNoteDuration},
            {"note_density",      TokenType::NoteDensity},
        };
        // Only emit attributes that are explicitly in track.attributes — matches
        // original encoder semantics where each Yellow/Expressive variant selects
        // which attributes to compute. Unset = not emitted.
        for (const auto& [name, type] : post_inst_attrs) {
            if (vocab_.has(type) && track.attributes.count(name)) {
                tokens.push_back(clamp_encode(type, track.attributes.at(name)));
            }
        }

        // Determine partial encoding for suffix-autoregressive mode
        bool is_partial = (static_cast<int>(track_idx) == opts.partial_encode_track_index)
                       && (opts.partial_encode_track_bars >= 0);
        int num_bars = is_partial
            ? std::min(opts.partial_encode_track_bars, static_cast<int>(track.bars.size()))
            : static_cast<int>(track.bars.size());

        // --- BARS ---
        for (int bar_idx = 0; bar_idx < num_bars; ++bar_idx) {
            const auto& bar = track.bars[bar_idx];

            if (vocab_.has(TokenType::Bar)) {
                tokens.push_back(vocab_.encode(TokenType::Bar, 0));
            }

            // Bar-level attribute tokens (tension, pitch class set, etc.)
            if (vocab_.has(TokenType::Tension) && track.attributes.count("bar_tension_" + std::to_string(bar_idx))) {
                tokens.push_back(clamp_encode(TokenType::Tension, track.attributes.at("bar_tension_" + std::to_string(bar_idx))));
            }
            if (vocab_.has(TokenType::PitchClassSet) && track.attributes.count("bar_pcs_" + std::to_string(bar_idx))) {
                tokens.push_back(clamp_encode(TokenType::PitchClassSet, track.attributes.at("bar_pcs_" + std::to_string(bar_idx))));
            }

            // TIME_SIGNATURE
            if (vocab_.has(TokenType::TimeSig)) {
                int mapped_ts = config.time_signatures
                    ? config.time_signatures->encode(bar.ts_numerator, bar.ts_denominator)
                    : 0;
                tokens.push_back(clamp_encode(TokenType::TimeSig, mapped_ts));
            }

            // Check for multi-fill placeholder
            bool is_infill = do_multi_fill
                && opts.multi_fill.count({static_cast<int>(track_idx), bar_idx});
            if (is_infill && vocab_.has(TokenType::FillInPlaceholder)) {
                tokens.push_back(vocab_.encode(TokenType::FillInPlaceholder, 0));
            }
            // Check if bar should be masked (future bars for lookahead)
            else if (bar.future && vocab_.has(TokenType::MaskBar)) {
                tokens.push_back(vocab_.encode(TokenType::MaskBar, 0));
            } else if (!is_infill) {
                encode_bar_notes(tokens, score, bar, is_drum, config, clamp_encode, vocab_);
            }

            if (vocab_.has(TokenType::BarEnd)) {
                tokens.push_back(vocab_.encode(TokenType::BarEnd, 0));
            }
        }

        // TRACK_END — omit for suffix-autoregressive partial encoding
        if (!is_partial && vocab_.has(TokenType::TrackEnd)) {
            tokens.push_back(vocab_.encode(TokenType::TrackEnd, 0));
        }
    }

    // --- MULTI-FILL BLOCKS ---
    if (do_multi_fill) {
        for (const auto& [t_idx, b_idx] : opts.multi_fill) {
            if (t_idx >= static_cast<int>(score.tracks.size())) continue;
            const auto& track = score.tracks[t_idx];
            if (b_idx >= static_cast<int>(track.bars.size())) continue;
            bool is_drum = (track.type == TrackType::Drum);

            if (vocab_.has(TokenType::FillInStart)) {
                tokens.push_back(vocab_.encode(TokenType::FillInStart, 0));
            }
            encode_bar_notes(tokens, score, track.bars[b_idx], is_drum, config, clamp_encode, vocab_);
            if (vocab_.has(TokenType::FillInEnd)) {
                tokens.push_back(vocab_.encode(TokenType::FillInEnd, 0));
            }
        }
    }

    // --- PIECE_END ---
    if (vocab_.has(TokenType::PieceEnd)) {
        tokens.push_back(vocab_.encode(TokenType::PieceEnd, 0));
    }
    return tokens;
}

} // namespace midigpt::tokenizer
