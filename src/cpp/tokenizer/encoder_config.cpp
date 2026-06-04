#include "encoder_config.h"
#include <nlohmann/json.hpp>
#include <set>
#include <stdexcept>

namespace midigpt::tokenizer {

static std::string to_string(TokenType t) {
    switch(t) {
        case TokenType::PieceStart: return "PieceStart";
        case TokenType::NoteOnset: return "NoteOnset";
        case TokenType::NoteOffset: return "NoteOffset";
        case TokenType::NotePitch: return "NotePitch";
        case TokenType::NonPitch: return "NonPitch";
        case TokenType::Velocity: return "Velocity";
        case TokenType::TimeDelta: return "TimeDelta";
        case TokenType::TimeAbsolutePos: return "TimeAbsolutePos";
        case TokenType::Instrument: return "Instrument";
        case TokenType::Bar: return "Bar";
        case TokenType::BarEnd: return "BarEnd";
        case TokenType::Track: return "Track";
        case TokenType::TrackEnd: return "TrackEnd";
        case TokenType::DrumTrack: return "DrumTrack";
        case TokenType::FillIn: return "FillIn";
        case TokenType::FillInPlaceholder: return "FillInPlaceholder";
        case TokenType::FillInStart: return "FillInStart";
        case TokenType::FillInEnd: return "FillInEnd";
        case TokenType::Header: return "Header";
        case TokenType::VelocityLevel: return "VelocityLevel";
        case TokenType::Genre: return "Genre";
        case TokenType::NoteDensity: return "NoteDensity";
        case TokenType::TimeSig: return "TimeSig";
        case TokenType::Segment: return "Segment";
        case TokenType::SegmentEnd: return "SegmentEnd";
        case TokenType::SegmentFillIn: return "SegmentFillIn";
        case TokenType::NoteDuration: return "NoteDuration";
        case TokenType::AvPolyphony: return "AvPolyphony";
        case TokenType::MinPolyphony: return "MinPolyphony";
        case TokenType::MaxPolyphony: return "MaxPolyphony";
        case TokenType::MinNoteDuration: return "MinNoteDuration";
        case TokenType::MaxNoteDuration: return "MaxNoteDuration";
        case TokenType::NumBars: return "NumBars";
        case TokenType::MinPolyphonyHard: return "MinPolyphonyHard";
        case TokenType::MaxPolyphonyHard: return "MaxPolyphonyHard";
        case TokenType::MinNoteDurationHard: return "MinNoteDurationHard";
        case TokenType::MaxNoteDurationHard: return "MaxNoteDurationHard";
        case TokenType::RestPercentage: return "RestPercentage";
        case TokenType::PitchClass: return "PitchClass";
        case TokenType::PitchClassCount: return "PitchClassCount";
        case TokenType::BarLevelOnsetDensity: return "BarLevelOnsetDensity";
        case TokenType::BarLevelOnsetPolyphonyMin: return "BarLevelOnsetPolyphonyMin";
        case TokenType::BarLevelOnsetPolyphonyMax: return "BarLevelOnsetPolyphonyMax";
        case TokenType::TrackLevelOnsetDensity: return "TrackLevelOnsetDensity";
        case TokenType::TrackLevelOnsetPolyphonyMin: return "TrackLevelOnsetPolyphonyMin";
        case TokenType::TrackLevelOnsetPolyphonyMax: return "TrackLevelOnsetPolyphonyMax";
        case TokenType::TrackLevelOnsetDensityMin: return "TrackLevelOnsetDensityMin";
        case TokenType::TrackLevelOnsetDensityMax: return "TrackLevelOnsetDensityMax";
        case TokenType::TrackLevelPitchRangeMin: return "TrackLevelPitchRangeMin";
        case TokenType::TrackLevelPitchRangeMax: return "TrackLevelPitchRangeMax";
        case TokenType::KeySignature: return "KeySignature";
        case TokenType::BarLevelPitchClassSet: return "BarLevelPitchClassSet";
        case TokenType::TrackLevelSilenceProportionMin: return "TrackLevelSilenceProportionMin";
        case TokenType::TrackLevelSilenceProportionMax: return "TrackLevelSilenceProportionMax";
        case TokenType::ValenceSpotify: return "ValenceSpotify";
        case TokenType::EnergySpotify: return "EnergySpotify";
        case TokenType::DanceabilitySpotify: return "DanceabilitySpotify";
        case TokenType::Danceability: return "Danceability";
        case TokenType::Tension: return "Tension";
        case TokenType::ContainsNoteDurationThirtySecond: return "ContainsNoteDurationThirtySecond";
        case TokenType::ContainsNoteDurationSixteenth: return "ContainsNoteDurationSixteenth";
        case TokenType::ContainsNoteDurationEighth: return "ContainsNoteDurationEighth";
        case TokenType::ContainsNoteDurationQuarter: return "ContainsNoteDurationQuarter";
        case TokenType::ContainsNoteDurationHalf: return "ContainsNoteDurationHalf";
        case TokenType::ContainsNoteDurationWhole: return "ContainsNoteDurationWhole";
        case TokenType::WnbdSyncopation: return "WnbdSyncopation";
        case TokenType::Repetition: return "Repetition";
        case TokenType::Delta: return "Delta";
        case TokenType::DeltaDirection: return "DeltaDirection";
        case TokenType::None: return "None";
        case TokenType::MaskBar: return "MaskBar";
        case TokenType::TensionDrum: return "TensionDrum";
        case TokenType::UseVelocity: return "UseVelocity";
        case TokenType::UseMicrotiming: return "UseMicrotiming";
        default: return "Unknown";
    }
}

static TokenType from_string(const std::string& s) {
    if (s == "PieceStart" || s == "TOKEN_PIECE_START") return TokenType::PieceStart;
    if (s == "NoteOnset" || s == "TOKEN_NOTE_ONSET") return TokenType::NoteOnset;
    if (s == "NoteOffset" || s == "TOKEN_NOTE_OFFSET") return TokenType::NoteOffset;
    if (s == "NotePitch" || s == "TOKEN_PITCH") return TokenType::NotePitch;
    if (s == "NonPitch" || s == "TOKEN_NON_PITCH") return TokenType::NonPitch;
    if (s == "Velocity" || s == "TOKEN_VELOCITY") return TokenType::Velocity;
    if (s == "TimeDelta" || s == "TOKEN_TIME_DELTA") return TokenType::TimeDelta;
    if (s == "TimeAbsolutePos" || s == "TOKEN_TIME_ABSOLUTE_POS") return TokenType::TimeAbsolutePos;
    if (s == "Instrument" || s == "TOKEN_INSTRUMENT") return TokenType::Instrument;
    if (s == "Bar" || s == "TOKEN_BAR") return TokenType::Bar;
    if (s == "BarEnd" || s == "TOKEN_BAR_END") return TokenType::BarEnd;
    if (s == "Track" || s == "TOKEN_TRACK") return TokenType::Track;
    if (s == "TrackEnd" || s == "TOKEN_TRACK_END") return TokenType::TrackEnd;
    if (s == "DrumTrack" || s == "TOKEN_DRUM_TRACK") return TokenType::DrumTrack;
    if (s == "FillIn" || s == "TOKEN_FILL_IN") return TokenType::FillIn;
    if (s == "FillInPlaceholder" || s == "TOKEN_FILL_IN_PLACEHOLDER") return TokenType::FillInPlaceholder;
    if (s == "FillInStart" || s == "TOKEN_FILL_IN_START") return TokenType::FillInStart;
    if (s == "FillInEnd" || s == "TOKEN_FILL_IN_END") return TokenType::FillInEnd;
    if (s == "Header" || s == "TOKEN_HEADER") return TokenType::Header;
    if (s == "VelocityLevel" || s == "TOKEN_VELOCITY_LEVEL") return TokenType::VelocityLevel;
    if (s == "Genre" || s == "TOKEN_GENRE") return TokenType::Genre;
    if (s == "NoteDensity" || s == "TOKEN_DENSITY_LEVEL") return TokenType::NoteDensity;
    if (s == "TimeSig" || s == "TOKEN_TIME_SIGNATURE") return TokenType::TimeSig;
    if (s == "Segment" || s == "TOKEN_SEGMENT") return TokenType::Segment;
    if (s == "SegmentEnd" || s == "TOKEN_SEGMENT_END") return TokenType::SegmentEnd;
    if (s == "SegmentFillIn" || s == "TOKEN_SEGMENT_FILL_IN") return TokenType::SegmentFillIn;
    if (s == "NoteDuration" || s == "TOKEN_NOTE_DURATION") return TokenType::NoteDuration;
    if (s == "AvPolyphony" || s == "TOKEN_AV_POLYPHONY") return TokenType::AvPolyphony;
    if (s == "MinPolyphony" || s == "TOKEN_MIN_POLYPHONY") return TokenType::MinPolyphony;
    if (s == "MaxPolyphony" || s == "TOKEN_MAX_POLYPHONY") return TokenType::MaxPolyphony;
    if (s == "MinNoteDuration" || s == "TOKEN_MIN_NOTE_DURATION") return TokenType::MinNoteDuration;
    if (s == "MaxNoteDuration" || s == "TOKEN_MAX_NOTE_DURATION") return TokenType::MaxNoteDuration;
    if (s == "NumBars" || s == "TOKEN_NUM_BARS") return TokenType::NumBars;
    if (s == "MinPolyphonyHard" || s == "TOKEN_MIN_POLYPHONY_HARD") return TokenType::MinPolyphonyHard;
    if (s == "MaxPolyphonyHard" || s == "TOKEN_MAX_POLYPHONY_HARD") return TokenType::MaxPolyphonyHard;
    if (s == "MinNoteDurationHard" || s == "TOKEN_MIN_NOTE_DURATION_HARD") return TokenType::MinNoteDurationHard;
    if (s == "MaxNoteDurationHard" || s == "TOKEN_MAX_NOTE_DURATION_HARD") return TokenType::MaxNoteDurationHard;
    if (s == "RestPercentage" || s == "TOKEN_REST_PERCENTAGE") return TokenType::RestPercentage;
    if (s == "PitchClass" || s == "TOKEN_PITCH_CLASS") return TokenType::PitchClass;
    if (s == "PitchClassCount" || s == "TOKEN_PITCH_CLASS_COUNT") return TokenType::PitchClassCount;
    if (s == "BarLevelOnsetDensity" || s == "TOKEN_BAR_LEVEL_ONSET_DENSITY") return TokenType::BarLevelOnsetDensity;
    if (s == "BarLevelOnsetPolyphonyMin" || s == "TOKEN_BAR_LEVEL_ONSET_POLYPHONY_MIN") return TokenType::BarLevelOnsetPolyphonyMin;
    if (s == "BarLevelOnsetPolyphonyMax" || s == "TOKEN_BAR_LEVEL_ONSET_POLYPHONY_MAX") return TokenType::BarLevelOnsetPolyphonyMax;
    if (s == "TrackLevelOnsetDensity" || s == "TOKEN_TRACK_LEVEL_ONSET_DENSITY") return TokenType::TrackLevelOnsetDensity;
    if (s == "TrackLevelOnsetPolyphonyMin" || s == "TOKEN_TRACK_LEVEL_ONSET_POLYPHONY_MIN") return TokenType::TrackLevelOnsetPolyphonyMin;
    if (s == "TrackLevelOnsetPolyphonyMax" || s == "TOKEN_TRACK_LEVEL_ONSET_POLYPHONY_MAX") return TokenType::TrackLevelOnsetPolyphonyMax;
    if (s == "TrackLevelOnsetDensityMin" || s == "TOKEN_TRACK_LEVEL_ONSET_DENSITY_MIN") return TokenType::TrackLevelOnsetDensityMin;
    if (s == "TrackLevelOnsetDensityMax" || s == "TOKEN_TRACK_LEVEL_ONSET_DENSITY_MAX") return TokenType::TrackLevelOnsetDensityMax;
    if (s == "TrackLevelPitchRangeMin" || s == "TOKEN_TRACK_LEVEL_PITCH_RANGE_MIN") return TokenType::TrackLevelPitchRangeMin;
    if (s == "TrackLevelPitchRangeMax" || s == "TOKEN_TRACK_LEVEL_PITCH_RANGE_MAX") return TokenType::TrackLevelPitchRangeMax;
    if (s == "KeySignature" || s == "TOKEN_KEY_SIGNATURE") return TokenType::KeySignature;
    if (s == "BarLevelPitchClassSet" || s == "TOKEN_BAR_LEVEL_PITCH_CLASS_SET") return TokenType::BarLevelPitchClassSet;
    if (s == "TrackLevelSilenceProportionMin" || s == "TOKEN_TRACK_LEVEL_SILENCE_PROPORTION_MIN") return TokenType::TrackLevelSilenceProportionMin;
    if (s == "TrackLevelSilenceProportionMax" || s == "TOKEN_TRACK_LEVEL_SILENCE_PROPORTION_MAX") return TokenType::TrackLevelSilenceProportionMax;
    if (s == "ValenceSpotify" || s == "TOKEN_VALENCE_SPOTIFY") return TokenType::ValenceSpotify;
    if (s == "EnergySpotify" || s == "TOKEN_ENERGY_SPOTIFY") return TokenType::EnergySpotify;
    if (s == "DanceabilitySpotify" || s == "TOKEN_DANCEABILITY_SPOTIFY") return TokenType::DanceabilitySpotify;
    if (s == "Danceability" || s == "TOKEN_DANCEABILITY") return TokenType::Danceability;
    if (s == "Tension" || s == "TOKEN_BAR_LEVEL_TENSION") return TokenType::Tension;
    if (s == "ContainsNoteDurationThirtySecond" || s == "TOKEN_CONTAINS_NOTE_DURATION_THIRTY_SECOND") return TokenType::ContainsNoteDurationThirtySecond;
    if (s == "ContainsNoteDurationSixteenth" || s == "TOKEN_CONTAINS_NOTE_DURATION_SIXTEENTH") return TokenType::ContainsNoteDurationSixteenth;
    if (s == "ContainsNoteDurationEighth" || s == "TOKEN_CONTAINS_NOTE_DURATION_EIGHTH") return TokenType::ContainsNoteDurationEighth;
    if (s == "ContainsNoteDurationQuarter" || s == "TOKEN_CONTAINS_NOTE_DURATION_QUARTER") return TokenType::ContainsNoteDurationQuarter;
    if (s == "ContainsNoteDurationHalf" || s == "TOKEN_CONTAINS_NOTE_DURATION_HALF") return TokenType::ContainsNoteDurationHalf;
    if (s == "ContainsNoteDurationWhole" || s == "TOKEN_CONTAINS_NOTE_DURATION_WHOLE") return TokenType::ContainsNoteDurationWhole;
    if (s == "WnbdSyncopation" || s == "TOKEN_WNBD_SYNCOPATION") return TokenType::WnbdSyncopation;
    if (s == "Repetition" || s == "TOKEN_REPETITION") return TokenType::Repetition;
    if (s == "Delta" || s == "TOKEN_DELTA") return TokenType::Delta;
    if (s == "DeltaDirection" || s == "TOKEN_DELTA_DIRECTION") return TokenType::DeltaDirection;
    if (s == "None" || s == "TOKEN_NONE") return TokenType::None;
    if (s == "MaskBar" || s == "TOKEN_MASK_BAR") return TokenType::MaskBar;
    if (s == "TensionDrum" || s == "TOKEN_BAR_LEVEL_TENSION_DRUM") return TokenType::TensionDrum;

    // Additional aliases for common refactor names
    if (s == "OnsetPolyphony") return TokenType::OnsetPolyphony;
    if (s == "PitchRange") return TokenType::PitchRange;
    if (s == "NoteDurationDist") return TokenType::NoteDurationDist;
    if (s == "SilenceProportion") return TokenType::SilenceProportion;
    if (s == "PitchClassSet") return TokenType::PitchClassSet;
    if (s == "PieceEnd") return TokenType::PieceEnd;
    if (s == "UseVelocity") return TokenType::UseVelocity;
    if (s == "UseMicrotiming") return TokenType::UseMicrotiming;

    throw std::runtime_error("Unknown token type string: " + s);
}

EncoderConfig EncoderConfig::from_json(const std::string& json_str) {
    auto j = nlohmann::json::parse(json_str);
    EncoderConfig c;
    if (j.contains("resolution")) c.resolution = j["resolution"];
    if (j.contains("decode_resolution")) c.decode_resolution = j["decode_resolution"];
    if (j.contains("emit_delta_tokens")) c.emit_delta_tokens = j["emit_delta_tokens"];
    if (j.contains("supports_infill")) c.supports_infill = j["supports_infill"];
    if (j.contains("supports_mask_bar_token")) c.supports_mask_bar_token = j["supports_mask_bar_token"];
    if (j.contains("velocity_sticky")) c.velocity_sticky = j["velocity_sticky"];
    if (j.contains("switchable_velocity"))    c.switchable_velocity    = j["switchable_velocity"];
    if (j.contains("switchable_microtiming")) c.switchable_microtiming = j["switchable_microtiming"];
    // switchable_microtiming implies emit_delta_tokens (model trained with deltas).
    if (c.switchable_microtiming) c.emit_delta_tokens = true;
    if (j.contains("pitch_range")) {
        auto pr = j["pitch_range"];
        if (!pr.is_array() || pr.size() != 2) {
            throw std::runtime_error("pitch_range must be [min, max]");
        }
        c.pitch_min = pr[0]; c.pitch_max = pr[1];
    }
    if (j.contains("velocity_levels"))         c.velocity_levels         = j["velocity_levels"];
    if (j.contains("note_duration_max_beats")) c.note_duration_max_beats = j["note_duration_max_beats"];
    if (j.contains("attribute_controls")) {
        c.attribute_controls_json = j["attribute_controls"].dump();
    }
    if (j.contains("time_signatures")) {
        c.time_signatures = TimeSignatureList::from_json(j["time_signatures"]);
    }
    if (j.contains("instrument_merge_groups")) {
        c.instrument_grouping = InstrumentGrouping::from_json(j["instrument_merge_groups"]);
    } else {
        // Identity mapping: every MIDI program is its own group (128 groups).
        c.instrument_grouping = InstrumentGrouping(std::vector<std::vector<int>>{}, 128);
    }
    if (j.contains("genre_groups")) {
        c.genre_grouping = GenreGrouping::from_json(j["genre_groups"]);
    }
    if (j.contains("num_bars_map")) {
        c.num_bars_map = ValueMapper(j["num_bars_map"].get<std::vector<int>>());
    }
    if (j.contains("token_domains")) {
        // Legacy / test fixture path — use explicit list verbatim.
        for (const auto& item : j["token_domains"]) {
            c.token_domains.push_back({
                from_string(item["type"]),
                item["domain_size"]
            });
        }
    } else {
        c.derive_token_domains();
    }
    return c;
}

static int ceil_div(int a, int b) { return (a + b - 1) / b; }

void EncoderConfig::derive_token_domains() {
    token_domains.clear();

    // Structural — always present.
    token_domains.push_back({TokenType::PieceStart, supports_infill ? 2 : 1});
    token_domains.push_back({TokenType::Track,      2});      // 0=melodic, 1=drums
    token_domains.push_back({TokenType::TrackEnd,   1});
    token_domains.push_back({TokenType::Bar,        1});
    token_domains.push_back({TokenType::BarEnd,     1});

    // Time signatures — number of supported sigs.
    if (time_signatures && time_signatures->size() > 0) {
        token_domains.push_back({TokenType::TimeSig, time_signatures->size()});
    }

    // Instruments — dense merge-group count.
    if (instrument_grouping) {
        token_domains.push_back({TokenType::Instrument, instrument_grouping->num_groups()});
    }

    // NumBars — discrete value set.
    if (num_bars_map && num_bars_map->size() > 0) {
        token_domains.push_back({TokenType::NumBars, static_cast<int>(num_bars_map->size())});
    }

    // Pitch (NoteOnset is the pitch token in this encoder).
    int pitch_domain = pitch_max - pitch_min + 1;
    token_domains.push_back({TokenType::NoteOnset, pitch_domain});

    // Note duration — domain = max_beats * resolution (linear, 1-tick granularity).
    token_domains.push_back({TokenType::NoteDuration, note_duration_max_beats * resolution});

    // Time-absolute-position — covers longest possible bar in ticks.
    // max_bar_ticks = ceil(max(num/den)) * 4 * resolution where 4 = quarters/whole.
    if (time_signatures && time_signatures->size() > 0) {
        int max_bar_ticks = 0;
        for (int i = 0; i < time_signatures->size(); ++i) {
            auto [n, d] = time_signatures->decode(i);
            // bar_ticks = n * (4 * resolution) / d; round up to be safe.
            int bt = ceil_div(n * 4 * resolution, d);
            if (bt > max_bar_ticks) max_bar_ticks = bt;
        }
        token_domains.push_back({TokenType::TimeAbsolutePos, max_bar_ticks});
    }

    // Piece-level switchable mode tokens — emitted right after PieceStart
    // when the model supports switching velocity / microtiming on or off.
    // Domain size 2: 0 = off, 1 = on.
    if (switchable_velocity)    token_domains.push_back({TokenType::UseVelocity,    2});
    if (switchable_microtiming) token_domains.push_back({TokenType::UseMicrotiming, 2});

    // Genre token — emitted after piece-level mode tokens when a grouping
    // is configured. Domain size = number of canonical genre labels.
    if (genre_grouping && genre_grouping->num_genres() > 0) {
        token_domains.push_back({TokenType::Genre, genre_grouping->num_genres()});
    }

    // Velocity quantization.
    token_domains.push_back({TokenType::VelocityLevel, velocity_levels});

    // Delta microtiming tokens — gated on emit_delta_tokens (also implied by switchable_microtiming).
    if (emit_delta_tokens) {
        token_domains.push_back({TokenType::DeltaDirection, 2});
        // half-resolution gives sub-tick precision for both ± directions
        token_domains.push_back({TokenType::Delta, std::max(1, resolution / 2)});
    }

    // Infill marker tokens.
    if (supports_infill) {
        token_domains.push_back({TokenType::FillInPlaceholder, 1});
        token_domains.push_back({TokenType::FillInStart,       1});
        token_domains.push_back({TokenType::FillInEnd,         1});
    }

    // MaskBar token — one of several ways to mask a bar (alternatives:
    // attention masking, omission). Only encoders whose vocab includes
    // this token can mask bars via the token method.
    if (supports_mask_bar_token) {
        token_domains.push_back({TokenType::MaskBar, 1});
    }

    // NOTE: attribute-control token domains are NOT derived here. They are
    // appended later by Python (which is the source of truth for attribute
    // class → token_type + size) via add_attribute_token_domains().
}

void EncoderConfig::add_attribute_token_domains(
    const std::vector<std::pair<std::string, int>>& specs) {
    // Idempotent: skip token types already in token_domains so this can be
    // safely called even when a legacy explicit token_domains list already
    // included the attribute slots.
    std::set<TokenType> existing;
    for (const auto& td : token_domains) existing.insert(td.type);
    for (const auto& [name, size] : specs) {
        TokenType t = from_string(name);
        if (existing.count(t)) continue;
        token_domains.push_back({t, size});
        existing.insert(t);
    }
}

std::string EncoderConfig::to_json() const {
    nlohmann::json j;
    j["resolution"]               = resolution;
    j["decode_resolution"]        = decode_resolution;
    j["emit_delta_tokens"]        = emit_delta_tokens;
    j["supports_infill"]          = supports_infill;
    j["supports_mask_bar_token"]  = supports_mask_bar_token;
    j["velocity_sticky"]          = velocity_sticky;
    j["switchable_velocity"]      = switchable_velocity;
    j["switchable_microtiming"]   = switchable_microtiming;
    j["pitch_range"]              = {pitch_min, pitch_max};
    j["velocity_levels"]          = velocity_levels;
    j["note_duration_max_beats"]  = note_duration_max_beats;
    try {
        j["attribute_controls"] = nlohmann::json::parse(attribute_controls_json);
    } catch (const std::exception&) {
        j["attribute_controls"] = nlohmann::json::array();
    }
    nlohmann::json td = nlohmann::json::array();
    for (const auto& d : token_domains) {
        td.push_back({
            {"type", to_string(d.type)},
            {"domain_size", d.domain_size}
        });
    }
    j["token_domains"] = td;
    if (num_bars_map) {
        j["num_bars_map"] = num_bars_map->values();
    }
    if (time_signatures) {
        j["time_signatures"] = time_signatures->to_json();
    }
    if (instrument_grouping) {
        j["instrument_merge_groups"] = instrument_grouping->to_json();
    }
    if (genre_grouping) {
        j["genre_groups"] = genre_grouping->to_json();
    }
    return j.dump(4);
}

} // namespace midigpt::tokenizer
