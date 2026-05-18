#pragma once

#include "constraint.h"

namespace midigpt::masking {

class DensityConstraint : public Constraint {
public:
    explicit DensityConstraint(int target_density) : target_density_(target_density) {}

    void step(int token, const tokenizer::Vocabulary& vocab) override;
    void apply(std::vector<bool>& mask, const tokenizer::Vocabulary& vocab) const override;

private:
    int target_density_;
    int note_count_ = 0;
};

} // namespace midigpt::masking
