#include "decoder.h"
#include "domain_transforms.h"
#include "../core/logging.h"
#include <algorithm>
#include <set>
#include <map>
#include <optional>

namespace midigpt::tokenizer {

Decoder::Decoder(const Vocabulary& vocab) : vocab_(vocab) {}

static std::vector<int> resolve_infill(const std::vector<int>& raw,
                                       const Vocabulary& vocab) {
    int ph = vocab.has(TokenType::FillInPlaceholder) ? vocab.encode(TokenType::FillInPlaceholder, 0) : -1;
    int fs = vocab.has(TokenType::FillInStart) ? vocab.encode(TokenType::FillInStart, 0) : -1;
    int fe = vocab.has(TokenType::FillInEnd) ? vocab.encode(TokenType::FillInEnd, 0) : -1;
    if (ph < 0 || fs < 0 || fe < 0) return raw;

    // Collect fill blocks: content between FILL_IN_START and FILL_IN_END
    std::vector<std::vector<int>> fills;
    for (size_t i = 0; i < raw.size(); ++i) {
        if (raw[i] == fs) {
            std::vector<int> block;
            for (size_t j = i + 1; j < raw.size() && raw[j] != fe; ++j) {
                block.push_back(raw[j]);
            }
            fills.push_back(std::move(block));
        }
    }

    // Rebuild: replace each FILL_IN_PLACEHOLDER with next fill block,
    // stop before first FILL_IN_START
    std::vector<int> out;
    size_t fill_idx = 0;
    for (size_t i = 0; i < raw.size(); ++i) {
        if (raw[i] == fs) break;
        if (raw[i] == ph && fill_idx < fills.size()) {
            out.insert(out.end(), fills[fill_idx].begin(), fills[fill_idx].end());
            fill_idx++;
        } else {
            out.push_back(raw[i]);
        }
    }
    return out;
}

Score Decoder::decode(const std::vector<int>& tokens) const {
    // Resolve multi-fill: replace FILL_IN_PLACEHOLDER with content from
    // FILL_IN_START/END blocks if present in the token stream.
    const std::vector<int>& resolved = resolve_infill(tokens, vocab_);

    Score score;
    score.resolution = vocab_.config().resolution;

    Track* current_track = nullptr;
    Bar* current_bar = nullptr;
    int current_time = 0;
    int current_velocity_level = 0;  // bin index, expanded via VelocityQuantizer
    std::optional<VelocityQuantizer> vel_quant;
    if (vocab_.has(TokenType::VelocityLevel)) {
        vel_quant.emplace(vocab_.domain_size(TokenType::VelocityLevel));
    }
    auto current_midi_velocity = [&]() -> int {
        if (vel_quant && current_velocity_level > 0) return vel_quant->decode(current_velocity_level);
        return 100;  // default if no VelocityLevel in vocab
    };
    int delta_direction = 1;
    int delta_total = 0;
    int beat_length = 4;
    int track_count = 0;
    int current_bar_idx = -1;  // -1 = before first bar in current track
    int last_token = -1;
    bool is_drum_track = false;

    for (int token : resolved) {
        auto [type, value] = vocab_.decode(token);

        switch (type) {

        case TokenType::Track: {
            // Flush previous bar/track if needed
            current_time = 0;
            delta_direction = 1;
            delta_total = 0;
            current_velocity_level = 0;
            current_bar_idx = -1;

            if (track_count >= static_cast<int>(score.tracks.size())) {
                score.tracks.emplace_back();
            }
            current_track = &score.tracks[track_count];

            // Track type token: 0 = melodic, 1 = drum.
            current_track->type = (value == 1) ? TrackType::Drum : TrackType::Melodic;
            is_drum_track = (current_track->type == TrackType::Drum);
            current_bar = nullptr;
            break;
        }

        case TokenType::TrackEnd:
            track_count++;
            current_track = nullptr;
            current_bar = nullptr;
            break;

        case TokenType::Instrument:
            if (current_track) {
                if (vocab_.config().instrument_grouping) {
                    current_track->instrument = vocab_.config().instrument_grouping->decode(value);
                } else {
                    current_track->instrument = value;
                }
            }
            break;

        case TokenType::Bar: {
            current_time = 0;
            delta_direction = 1;
            delta_total = 0;
            beat_length = 4; // default, overridden by TimeSig
            current_bar_idx++;

            if (current_track) {
                current_track->bars.emplace_back();
                current_bar = &current_track->bars.back();
            }
            break;
        }

        case TokenType::TimeSig: {
            if (vocab_.config().time_signatures) {
                auto [num, den] = vocab_.config().time_signatures->decode(value);
                beat_length = 4 * num / den;
                if (current_bar) {
                    current_bar->ts_numerator = num;
                    current_bar->ts_denominator = den;
                }
            }
            break;
        }

        case TokenType::BarEnd: {
            if (current_bar) {
                int bar_ticks = beat_length * score.resolution;
                current_bar->beat_length = beat_length;
                current_time = bar_ticks;
            }
            break;
        }

        case TokenType::TimeAbsolutePos:
            current_time = value;
            delta_direction = 1;
            delta_total = 0;
            break;

        case TokenType::VelocityLevel:
            current_velocity_level = value;
            break;

        case TokenType::DeltaDirection:
            delta_direction = -1;
            delta_total = 0;
            break;

        case TokenType::Delta:
            delta_total += delta_direction * value;
            break;

        case TokenType::NoteOnset:
        case TokenType::NotePitch: {
            if (current_bar && current_track) {
                if (is_drum_track) {
                    // Drum: create note immediately with duration=1
                    Note n;
                    n.pitch = value;
                    n.velocity = current_midi_velocity();
                    n.onset_ticks = current_time;
                    n.duration_ticks = 1;
                    n.delta = delta_total;
                    delta_total = 0;
                    delta_direction = 1;

                    int note_idx = static_cast<int>(score.notes.size());
                    score.notes.push_back(n);
                    current_bar->note_indices.push_back(note_idx);
                    current_bar->has_notes = true;
                }
                // For melodic: pitch is recorded via last_token,
                // note creation happens on NoteDuration
            }
            break;
        }

        case TokenType::NoteDuration: {
            // Dual role: attribute control (NoteDurationDist = 26) before the
            // first Bar in a track (current_bar_idx < 0), or note duration
            // token inside a bar. Position disambiguates.
            if (current_track && current_bar_idx < 0) {
                current_track->attributes["note_duration_dist"] = value;
            } else if (current_bar && current_track && last_token >= 0) {
                auto [lt_type, lt_value] = vocab_.decode(last_token);
                if (lt_type == TokenType::NoteOnset || lt_type == TokenType::NotePitch) {
                    Note n;
                    n.pitch = lt_value;
                    n.velocity = current_midi_velocity();
                    n.onset_ticks = current_time;
                    n.duration_ticks = value + 1; // +1 offset (original convention)
                    n.delta = delta_total;
                    delta_total = 0;
                    delta_direction = 1;

                    int note_idx = static_cast<int>(score.notes.size());
                    score.notes.push_back(n);
                    current_bar->note_indices.push_back(note_idx);
                    current_bar->has_notes = true;
                }
            }
            break;
        }

        case TokenType::UseVelocity:
        case TokenType::UseMicrotiming:
            // Piece-level mode tokens — recorded for round-trip but do not
            // affect note decoding (velocity/delta are already in the stream
            // when present; absence means the mode was off).
            break;

        case TokenType::MaskBar:
            // Masked bar — no notes to decode
            break;

        case TokenType::FillInPlaceholder:
        case TokenType::FillInStart:
        case TokenType::FillInEnd:
            // Infill tokens — handled at a higher level
            break;

        // Attribute controls: preserve into track.attributes so encode→decode→encode
        // is a fixed point.
        //
        // Track-level: key = Python attribute name (e.g. "min_polyphony").
        // Bar-level:   key = "bar_{token_type_string}_{bar_idx}" matching the
        //              format used by AttributeAnalyzer.compute_bar_tokens().
        //
        // Token 42 (BarLevelOnsetPolyphonyMax / OnsetPolyphony) is disambiguated
        // by position: track-level attrs appear before any Bar token
        // (current_bar_idx == -1); bar-level appear after.

        case TokenType::NoteDensity:
            if (current_track) current_track->attributes["note_density"] = value;
            break;
        case TokenType::MinPolyphony:
            if (current_track) current_track->attributes["min_polyphony"] = value;
            break;
        case TokenType::MaxPolyphony:
            if (current_track) current_track->attributes["max_polyphony"] = value;
            break;
        case TokenType::MinNoteDuration:
            if (current_track) current_track->attributes["min_note_duration"] = value;
            break;
        case TokenType::MaxNoteDuration:
            if (current_track) current_track->attributes["max_note_duration"] = value;
            break;
        case TokenType::SilenceProportion:   // = TrackLevelSilenceProportionMax = 53
            if (current_track) current_track->attributes["silence_proportion"] = value;
            break;
        case TokenType::KeySignature:
            if (current_track) current_track->attributes["key_signature"] = value;
            break;
        case TokenType::PitchRange:          // = TrackLevelPitchRangeMax = 49
            if (current_track) current_track->attributes["pitch_range"] = value;
            break;

        // Token 42: BarLevelOnsetPolyphonyMax == OnsetPolyphony.
        // Position disambiguates: before first Bar → track-level "onset_polyphony";
        // after a Bar token → bar-level "bar_BarLevelOnsetPolyphonyMax_N".
        case TokenType::BarLevelOnsetPolyphonyMax:
            if (current_track) {
                if (current_bar_idx < 0) {
                    current_track->attributes["onset_polyphony"] = value;
                } else {
                    current_track->attributes[
                        "bar_BarLevelOnsetPolyphonyMax_" + std::to_string(current_bar_idx)
                    ] = value;
                }
            }
            break;

        case TokenType::BarLevelOnsetDensity:
            if (current_track && current_bar_idx >= 0)
                current_track->attributes[
                    "bar_BarLevelOnsetDensity_" + std::to_string(current_bar_idx)
                ] = value;
            break;

        case TokenType::BarLevelOnsetPolyphonyMin:
            if (current_track && current_bar_idx >= 0)
                current_track->attributes[
                    "bar_BarLevelOnsetPolyphonyMin_" + std::to_string(current_bar_idx)
                ] = value;
            break;

        case TokenType::BarLevelPitchClassSet:  // = PitchClassSet = 51
            if (current_track && current_bar_idx >= 0)
                current_track->attributes[
                    "bar_PitchClassSet_" + std::to_string(current_bar_idx)
                ] = value;
            break;

        case TokenType::TrackLevelSilenceProportionMin:
            if (current_track) current_track->attributes["silence_proportion_min"] = value;
            break;

        case TokenType::TrackLevelOnsetDensity:
            if (current_track) current_track->attributes["track_onset_density"] = value;
            break;
        case TokenType::TrackLevelOnsetPolyphonyMin:
            if (current_track) current_track->attributes["track_onset_polyphony_min"] = value;
            break;

        default:
            // PieceStart, PieceEnd, NumBars — skip
            break;
        }

        last_token = token;
    }

    // Flush trailing track if no TRACK_END was seen (suffix-AR mode)
    if (current_track && track_count < static_cast<int>(score.tracks.size())) {
        track_count++;
    }

    return score;
}

} // namespace midigpt::tokenizer
