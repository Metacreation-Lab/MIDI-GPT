#pragma once

#include "constraint.h"

namespace midigpt::masking {

class PolyphonyConstraint : public Constraint {
public:
    explicit PolyphonyConstraint(int max_polyphony) : max_polyphony_(max_polyphony) {}

    void step(int token, const tokenizer::Vocabulary& vocab) override;
    void apply(std::vector<bool>& mask, const tokenizer::Vocabulary& vocab) const override;

private:
    int max_polyphony_;
    int onset_count_ = 0;
};

} // namespace midigpt::masking
