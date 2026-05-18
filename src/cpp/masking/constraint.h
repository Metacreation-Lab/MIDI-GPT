#pragma once

#include <vector>
#include "../tokenizer/vocabulary.h"

namespace midigpt::masking {

class Constraint {
public:
    virtual ~Constraint() = default;

    // Updates internal state based on a generated token
    virtual void step(int token, const tokenizer::Vocabulary& vocab) = 0;

    // Fills the given mask (size = vocab_size) with true for disallowed tokens
    // Implementations should OR their restrictions into the mask
    virtual void apply(std::vector<bool>& mask, const tokenizer::Vocabulary& vocab) const = 0;

    // Optional: notify the constraint that the current/next FillIn block targets
    // a track of the given drumness. Default no-op; overridden by GrammarConstraint
    // so it can require NoteDuration after NoteOnset for melodic tracks even when
    // the most recent Track token in the prompt context was a drum track.
    virtual void set_fillin_drum(bool /*is_drum*/) {}
};

} // namespace midigpt::masking
