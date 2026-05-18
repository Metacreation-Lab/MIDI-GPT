#pragma once

#include "vocabulary.h"
#include "../core/score.h"
#include <vector>

namespace midigpt::tokenizer {

class Decoder {
public:
    explicit Decoder(const Vocabulary& vocab);

    // Decodes a linear sequence of tokens back into a Score structure
    Score decode(const std::vector<int>& tokens) const;

private:
    const Vocabulary& vocab_;
};

} // namespace midigpt::tokenizer
