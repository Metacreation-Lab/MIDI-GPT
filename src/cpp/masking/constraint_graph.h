#pragma once

#include "constraint.h"
#include <memory>

namespace midigpt::masking {

class ConstraintGraph {
public:
    ConstraintGraph() = default;

    void add_constraint(std::shared_ptr<Constraint> constraint);

    void step(int token, const tokenizer::Vocabulary& vocab);

    // Notify all constraints that the current FillIn block targets a track
    // of the given drumness (see Constraint::set_fillin_drum).
    void set_fillin_drum(bool is_drum);

    // Returns a full boolean mask where true = masked (disallowed)
    std::vector<bool> get_mask(const tokenizer::Vocabulary& vocab) const;

private:
    std::vector<std::shared_ptr<Constraint>> constraints_;
};

} // namespace midigpt::masking
