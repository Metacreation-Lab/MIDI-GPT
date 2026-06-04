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
        const tokenizer::Decoder&                 decoder,
        bool                                      use_span_masks = false,
        bool                                      remove_future_bars = false,
        int                                       use_velocity = -1,
        int                                       use_microtiming = -1,
        int                                       genre = -1
    );

    bool              complete()       const; // all bars_to_generate are done
    std::vector<int>  context_tokens() const; // full context for model forward
    std::vector<bool> logit_mask()     const; // from ConstraintGraph — valid next tokens
    void              advance(int token);     // append token, update internal state
    Score             result()         const; // decode + apply generated bars into context

    // [start,end) token-index ranges that should be hidden from self-attention
    // (set only when constructed with use_span_masks=true). Indexed into the
    // sequence returned by context_tokens(); generated tokens are never hidden.
    std::vector<std::pair<int,int>> hidden_spans() const { return hidden_spans_; }

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

    // Bars OUTSIDE the step's window, saved per track (in original track
    // order) before the trim/shift mutations. result() splices them back
    // around the decoded window so the returned Score uses absolute bar
    // indices — callers see the full piece, not just the window. note_indices
    // on these bars reference original_notes_, NOT the decoded result's pool;
    // result() copies the notes across and remaps indices on the fly.
    std::vector<std::vector<Bar>>  prefix_bars_;   // bars [0, original_start_bar)
    std::vector<std::vector<Bar>>  suffix_bars_;   // bars [original_end_bar, ...)
    std::vector<Note>              original_notes_;
    int                            original_start_bar_ = 0;

    // Token-index ranges within context_cache_ that should be attention-masked.
    // Populated by the encoder when use_span_masks is enabled.
    std::vector<std::pair<int,int>> hidden_spans_;

    bool               remove_future_bars_ = false;
};

} // namespace midigpt::sampling
