#pragma once

#include "selection_mask.h"
#include "generation_step.h"
#include "../tokenizer/encoder_config.h"

namespace midigpt::sampling {

class StepPlanner {
public:
    // bars_per_step=0 means "default to model_dim"
    StepPlanner(const SelectionMask& mask, const tokenizer::EncoderConfig& config,
                int bars_per_step = 0, int tracks_per_step = 1);
    std::vector<GenerationStep> plan() const;

private:
    void find_steps_inner(std::vector<GenerationStep>& steps,
                          std::vector<std::vector<bool>>& generated,
                          bool autoregressive) const;

    SelectionMask mask_;
    tokenizer::EncoderConfig config_;
    int bars_per_step_;
    int tracks_per_step_;
};

} // namespace midigpt::sampling
