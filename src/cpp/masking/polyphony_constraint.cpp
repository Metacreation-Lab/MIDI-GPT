#include "polyphony_constraint.h"

namespace midigpt::masking {

void PolyphonyConstraint::step(int token, const tokenizer::Vocabulary& vocab) {
    TokenType type = vocab.get_type(token);
    if (type == TokenType::Bar || type == TokenType::BarEnd) {
        onset_count_ = 0;
    } else if (type == TokenType::TimeAbsolutePos) {
        onset_count_ = 0;
    } else if (type == TokenType::NoteOnset || type == TokenType::NotePitch) {
        onset_count_++;
    }
}

void PolyphonyConstraint::apply(std::vector<bool>& mask, const tokenizer::Vocabulary& vocab) const {
    if (onset_count_ >= max_polyphony_) {
        auto [start, end] = vocab.range(TokenType::NoteOnset);
        if (start != -1) {
            for (int i = start; i < end; ++i) {
                mask[i] = true;
            }
        }
        auto [ps, pe] = vocab.range(TokenType::NotePitch);
        if (ps != -1) {
            for (int i = ps; i < pe; ++i) {
                mask[i] = true;
            }
        }
    }
}

} // namespace midigpt::masking
