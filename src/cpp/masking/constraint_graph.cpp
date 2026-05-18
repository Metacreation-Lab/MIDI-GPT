#include "constraint_graph.h"

namespace midigpt::masking {

void ConstraintGraph::add_constraint(std::shared_ptr<Constraint> constraint) {
    if (constraint) {
        constraints_.push_back(constraint);
    }
}

void ConstraintGraph::step(int token, const tokenizer::Vocabulary& vocab) {
    for (auto& constraint : constraints_) {
        constraint->step(token, vocab);
    }
}

void ConstraintGraph::set_fillin_drum(bool is_drum) {
    for (auto& constraint : constraints_) {
        constraint->set_fillin_drum(is_drum);
    }
}

std::vector<bool> ConstraintGraph::get_mask(const tokenizer::Vocabulary& vocab) const {
    std::vector<bool> mask(vocab.size(), false);
    for (const auto& constraint : constraints_) {
        constraint->apply(mask, vocab);
    }
    return mask;
}

} // namespace midigpt::masking
