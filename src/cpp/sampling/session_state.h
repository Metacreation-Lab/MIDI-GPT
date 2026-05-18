#pragma once

#include "generation_step.h"
#include "../core/score.h"
#include "../tokenizer/vocabulary.h"
#include "../tokenizer/encoder.h"
#include "../tokenizer/decoder.h"
#include "../masking/constraint_graph.h"
#include <vector>

namespace midigpt::sampling {

class SessionState {
public:
    SessionState(
        Score                                     context,
        const GenerationStep&                     step,
        const tokenizer::Vocabulary&              vocab,
        const masking::ConstraintGraph&           constraints,
        const tokenizer::Encoder&                 encoder,
        const tokenizer::Decoder&                 decoder
    );

    bool              complete()       const; // all bars_to_generate are done
    std::vector<int>  context_tokens() const; // full context for model forward
    std::vector<bool> logit_mask()     const; // from ConstraintGraph — valid next tokens
    void              advance(int token);     // append token, update internal state
    Score             result()         const; // decode + apply generated bars into context

private:
    Score              context_;
    GenerationStep     step_;
    const tokenizer::Vocabulary&  vocab_;
    masking::ConstraintGraph      constraints_;      // owns a copy — constraints are step-local
    const tokenizer::Encoder&     encoder_;
    const tokenizer::Decoder&     decoder_;
    std::vector<int>   generated_;       // tokens produced so far in this step
    std::vector<int>   context_cache_;   // pre-encoded context (immutable during step)

    // Ordered drum flags for each FillIn block, matching encoder iteration of
    // GenerationStep::bars_to_generate (std::set lexicographic order). Used to
    // tell GrammarConstraint whether the *current* fill target is a drum track,
    // since the constraint can't infer it from the last Track token in context.
    std::vector<bool>  fillin_drum_order_;
    size_t             fillin_idx_ = 0;  // next index into fillin_drum_order_

    // For autoregressive mode: the agent track is moved to the last position
    // before encoding so its Track + Instrument tokens end the prompt. This
    // records the agent's original position so result() can restore the
    // user-visible track order.
    int                ar_original_agent_idx_ = -1;
};

} // namespace midigpt::sampling
