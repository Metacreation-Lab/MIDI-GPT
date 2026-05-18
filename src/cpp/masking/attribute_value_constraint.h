#pragma once

#include "constraint.h"

namespace midigpt::masking {

class AttributeValueConstraint : public Constraint {
public:
    AttributeValueConstraint(TokenType target_type, int allowed_value) 
        : target_type_(target_type), allowed_value_(allowed_value) {}

    void step(int /*token*/, const tokenizer::Vocabulary& /*vocab*/) override {}

    void apply(std::vector<bool>& mask, const tokenizer::Vocabulary& vocab) const override {
        auto [start, end] = vocab.range(target_type_);
        if (start != -1) {
            for (int i = start; i < end; ++i) {
                // mask out all values except the allowed one
                if (i - start != allowed_value_) {
                    mask[i] = true;
                }
            }
        }
    }

private:
    TokenType target_type_;
    int allowed_value_;
};

} // namespace midigpt::masking
