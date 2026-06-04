#pragma once

#include <vector>
#include <string>
#include <optional>
#include "../core/types.h"
#include "domain_transforms.h"

namespace midigpt::tokenizer {

struct TokenDomain {
    TokenType type;
    int       domain_size;
};

struct EncoderConfig {
    // Core
    int  resolution        = 480;   // ticks per beat (quarter note)
    int  decode_resolution = 480;   // target resolution after resample (= resolution → no resample)
    int  model_dim         = 4;     // runtime bars-per-step hint (step planner)

    // Token-domain primitives
    int  pitch_min               = 0;     // inclusive MIDI pitch lower bound (NoteOnset domain)
    int  pitch_max               = 127;   // inclusive MIDI pitch upper bound
    int  velocity_levels         = 32;    // # VelocityLevel quantization bins
    int  note_duration_max_beats = 8;     // max NoteDuration in beats (domain = max_beats * resolution)

    // Capability flags
    bool emit_delta_tokens       = false; // emit Delta / DeltaDirection (microtiming via Δ tokens)
    bool supports_infill         = false; // model can do FILL_IN_* tokens
    bool supports_mask_bar_token = false; // vocab includes MaskBar token
    bool velocity_sticky         = true;  // emit VELOCITY only when it changes (else per-note)
    // Switchable modes: model was trained to handle both on and off.
    // Setting either to true implies the feature is also on by default
    // (switchable_microtiming=true forces emit_delta_tokens=true).
    bool switchable_velocity    = false;
    bool switchable_microtiming = false;

    // Structured list of attribute controls this encoder exposes, as a
    // raw JSON fragment (default "[]"). The Python AttributeAnalyzer reads
    // this via a name → class registry. Each entry must include
    // {"name", "token_type", "size"}; C++ uses (token_type, size) for the
    // vocab, Python cross-validates size against the registered class.
    std::string attribute_controls_json = "[]";

    // Derived: filled in by derive_token_domains() based on the primitives
    // and the attribute_controls_json list. Tests may also push directly.
    std::vector<TokenDomain> token_domains;

    // Domain transforms — configurable per model
    std::optional<TimeSignatureList>  time_signatures;
    std::optional<InstrumentGrouping> instrument_grouping;
    std::optional<GenreGrouping>      genre_grouping;
    std::optional<ValueMapper>        num_bars_map;

    static EncoderConfig from_json(const std::string& json_str);
    std::string          to_json()                          const;

    // Populate token_domains from the primitive fields. Structural tokens
    // only — attribute-control token domains are appended later via
    // add_attribute_token_domains(), since their token_type/size live in
    // the Python attribute classes (single source of truth).
    void derive_token_domains();

    // Append (token_type_name, size) pairs to token_domains. Called by
    // Python after instantiating attribute classes.
    void add_attribute_token_domains(
        const std::vector<std::pair<std::string, int>>& specs);
};

} // namespace midigpt::tokenizer
