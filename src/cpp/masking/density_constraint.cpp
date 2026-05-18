#include "density_constraint.h"

namespace midigpt::masking {

void DensityConstraint::step(int token, const tokenizer::Vocabulary& vocab) {
    TokenType type = vocab.get_type(token);
    if (type == TokenType::Bar || type == TokenType::BarEnd) {
        note_count_ = 0;
    } else if (type == TokenType::NoteOnset || type == TokenType::NotePitch) {
        note_count_++;
    }
}

void DensityConstraint::apply(std::vector<bool>& mask, const tokenizer::Vocabulary& vocab) const {
    if (note_count_ >= target_density_) {
        auto block = [&](TokenType type) {
            auto [start, end] = vocab.range(type);
            if (start != -1) {
                for (int i = start; i < end; ++i) mask[i] = true;
            }
        };
        block(TokenType::NoteOnset);
        block(TokenType::NotePitch);
        block(TokenType::VelocityLevel);
        block(TokenType::TimeAbsolutePos);
        block(TokenType::DeltaDirection);
        block(TokenType::Delta);
    }
}

} // namespace midigpt::masking
