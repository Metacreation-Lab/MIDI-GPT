#include "session_state.h"
#include "../core/logging.h"

namespace midigpt::sampling {

SessionState::SessionState(
    Score                                     context,
    const GenerationStep&                     step,
    const tokenizer::Vocabulary&              vocab,
    const masking::ConstraintGraph&           constraints,
    const tokenizer::Encoder&                 encoder,
    const tokenizer::Decoder&                 decoder,
    bool                                      use_span_masks
) : context_(std::move(context)), step_(step), vocab_(vocab),
    constraints_(constraints), encoder_(encoder), decoder_(decoder)
{
    // Snapshot bars OUTSIDE the step window before any trim/shift, so result()
    // can splice them back and return a Score with absolute bar indices.
    // Saved in original track order — the AR reorder below permutes context_
    // but result() reverses that before splicing. The bars carry note_indices
    // that point into the ORIGINAL notes pool, so we keep a copy of that pool
    // and remap indices at splice time into the decoded result's pool.
    original_start_bar_ = step_.start_bar;
    original_notes_     = context_.notes;
    prefix_bars_.resize(context_.tracks.size());
    suffix_bars_.resize(context_.tracks.size());
    for (size_t ti = 0; ti < context_.tracks.size(); ++ti) {
        const auto& bars = context_.tracks[ti].bars;
        int n = static_cast<int>(bars.size());
        int s = std::min(step_.start_bar, n);
        int e = std::min(step_.end_bar, n);
        prefix_bars_[ti].assign(bars.begin(), bars.begin() + s);
        if (e < n) suffix_bars_[ti].assign(bars.begin() + e, bars.end());
    }

    // Trim bars to the step's window [start_bar, end_bar)
    for (auto& track : context_.tracks) {
        if (static_cast<int>(track.bars.size()) > step_.end_bar) {
            track.bars.resize(step_.end_bar);
        }
    }

    // Apply sliding window: drop bars before start_bar so the model only
    // sees model_dim bars, preventing context overflow on long sessions.
    if (step_.start_bar > 0) {
        int shift = step_.start_bar;
        for (auto& track : context_.tracks) {
            if (shift <= static_cast<int>(track.bars.size())) {
                track.bars.erase(track.bars.begin(), track.bars.begin() + shift);
            } else {
                track.bars.clear();
            }
        }
        std::set<std::pair<int,int>> new_btg;
        for (const auto& [tr, br] : step_.bars_to_generate) {
            new_btg.insert({tr, br - shift});
        }
        step_.bars_to_generate = std::move(new_btg);
        for (auto& row : step_.context) {
            if (shift <= static_cast<int>(row.size())) {
                row.erase(row.begin(), row.begin() + shift);
            } else {
                row.clear();
            }
        }
        for (auto& tup : step_.bar_mapping) {
            std::get<3>(tup) -= shift;
        }
        step_.end_bar -= shift;
        step_.start_bar = 0;
    }

    tokenizer::EncodeOptions encode_opts;
    encode_opts.window_bars = step_.end_bar - step_.start_bar;
    encode_opts.use_span_masks = use_span_masks;

    if (step_.is_autoregressive) {
        // --- Autoregressive step: suffix-AR encoding ---
        // Find the agent track (has bars_to_generate).
        int agent_track = -1;
        for (const auto& [tr, br] : step_.bars_to_generate) {
            if (agent_track < 0 || tr == agent_track) {
                agent_track = tr;
            }
        }

        // Reorder: move the agent track to the LAST position so its
        // Track + Instrument tokens end the encoded prompt. This lets the
        // model continue directly into attribute slots / bars / notes for the
        // agent track, and keeps GrammarConstraint::is_drum_ correctly set to
        // the agent's drumness (the last Track token wins).
        if (agent_track >= 0
            && agent_track != static_cast<int>(context_.tracks.size()) - 1) {
            ar_original_agent_idx_ = agent_track;
            int new_last = static_cast<int>(context_.tracks.size()) - 1;

            // Permute tracks.
            auto agent_track_obj = std::move(context_.tracks[agent_track]);
            context_.tracks.erase(context_.tracks.begin() + agent_track);
            context_.tracks.push_back(std::move(agent_track_obj));

            // Index remap: agent -> new_last; indices > agent shift down by 1.
            auto remap = [&](int old_idx) {
                if (old_idx == agent_track) return new_last;
                if (old_idx > agent_track) return old_idx - 1;
                return old_idx;
            };

            std::set<std::pair<int,int>> new_btg;
            for (const auto& [tr, br] : step_.bars_to_generate) {
                new_btg.insert({remap(tr), br});
            }
            step_.bars_to_generate = std::move(new_btg);

            for (auto& idx : step_.track_indices) idx = remap(idx);

            for (auto& tup : step_.bar_mapping) {
                std::get<2>(tup) = remap(std::get<2>(tup));
            }

            if (static_cast<int>(step_.context.size()) > agent_track) {
                auto agent_ctx = std::move(step_.context[agent_track]);
                step_.context.erase(step_.context.begin() + agent_track);
                step_.context.push_back(std::move(agent_ctx));
            }

            agent_track = new_last;
        }

        if (agent_track >= 0) {
            int first_gen_bar = step_.end_bar;
            for (const auto& [tr, br] : step_.bars_to_generate) {
                if (tr == agent_track && br < first_gen_bar) {
                    first_gen_bar = br;
                }
            }
            encode_opts.partial_encode_track_index = agent_track;
            encode_opts.partial_encode_track_bars = first_gen_bar;
        }

        // Apply bar masking: bars in the window that are context=false and not
        // being generated should be marked as future (produces MaskBar tokens)
        for (int ti = 0; ti < static_cast<int>(context_.tracks.size()); ++ti) {
            for (int bi = step_.start_bar; bi < step_.end_bar && bi < static_cast<int>(context_.tracks[ti].bars.size()); ++bi) {
                bool is_context = (ti < static_cast<int>(step_.context.size())
                                   && bi < static_cast<int>(step_.context[ti].size())
                                   && step_.context[ti][bi]);
                bool is_gen = step_.bars_to_generate.count({ti, bi}) > 0;
                if (!is_context && !is_gen) {
                    context_.tracks[ti].bars[bi].future = true;
                }
            }
        }

    } else {
        // --- Infill step: multi-fill encoding ---
        // The encoder will emit FILL_IN_PLACEHOLDER for bars in multi_fill,
        // then append FILL_IN_START/content/FILL_IN_END blocks.
        for (const auto& [tr, br] : step_.bars_to_generate) {
            encode_opts.multi_fill.insert({tr, br});
        }
        // Build the drum-flag order for FillIn blocks, matching the encoder's
        // iteration over encode_opts.multi_fill (std::set lexicographic order).
        fillin_drum_order_.clear();
        for (const auto& [tr, br] : encode_opts.multi_fill) {
            bool is_drum = (tr >= 0
                            && tr < static_cast<int>(context_.tracks.size())
                            && context_.tracks[tr].type == TrackType::Drum);
            fillin_drum_order_.push_back(is_drum);
        }

        // Apply bar masking for non-context bars
        for (int ti = 0; ti < static_cast<int>(context_.tracks.size()); ++ti) {
            for (int bi = step_.start_bar; bi < step_.end_bar && bi < static_cast<int>(context_.tracks[ti].bars.size()); ++bi) {
                bool is_context = (ti < static_cast<int>(step_.context.size())
                                   && bi < static_cast<int>(step_.context[ti].size())
                                   && step_.context[ti][bi]);
                bool is_gen = step_.bars_to_generate.count({ti, bi}) > 0;
                if (!is_context && !is_gen) {
                    context_.tracks[ti].bars[bi].future = true;
                }
            }
        }
    }

    // Encode the windowed context using the shared encoder + per-step options.
    {
        tokenizer::EncodeResult enc = encoder_.encode_full(context_, encode_opts);
        context_cache_ = std::move(enc.tokens);
        hidden_spans_  = std::move(enc.hidden_spans);
    }

    if (step_.is_autoregressive) {
        // Strip trailing structural tokens so the model can continue generating.
        //
        // The agent's target bars are empty in the encoded prompt, each rendered
        // as `Bar TimeSig BarEnd`. We want the prompt to END with the first
        // target bar's `Bar TimeSig` exposed — that way the model autoregresses
        // notes for bar 1, emits BarEnd, then (if N>1) opens further `Bar TimeSig
        // ... BarEnd` blocks for bars 2..N, then TrackEnd.
        //
        // Pop in this order: trailing TrackEnd/PieceEnd, then for each of the
        // last (N-1) target bars pop a full `BarEnd TimeSig Bar` (right-to-left),
        // then finally pop the first target bar's `BarEnd` — leaving its
        // `Bar TimeSig` intact.
        auto pop_if = [&](TokenType t) -> bool {
            if (!context_cache_.empty()
                && vocab_.get_type(context_cache_.back()) == t) {
                context_cache_.pop_back();
                return true;
            }
            return false;
        };
        while (pop_if(TokenType::PieceEnd) || pop_if(TokenType::TrackEnd)) {}
        int N = static_cast<int>(step_.bars_to_generate.size());
        for (int i = 0; i < N - 1; ++i) {
            if (!pop_if(TokenType::BarEnd)) break;
            if (!pop_if(TokenType::TimeSig)) break;
            if (!pop_if(TokenType::Bar))     break;
        }
        pop_if(TokenType::BarEnd);
    } else {
        // For infill: truncate at the first FILL_IN_START token
        // The model generates fill blocks from this point
        int fill_start_token = -1;
        if (vocab_.has(TokenType::FillInStart)) {
            fill_start_token = vocab_.encode(TokenType::FillInStart, 0);
        }
        if (fill_start_token >= 0) {
            for (size_t idx = 0; idx < context_cache_.size(); ++idx) {
                if (context_cache_[idx] == fill_start_token) {
                    context_cache_.resize(idx + 1); // keep the FILL_IN_START token
                    break;
                }
            }
        }
    }

    // Trim hidden spans to the (possibly popped) cache and drop empty ones.
    if (!hidden_spans_.empty()) {
        const int cache_len = static_cast<int>(context_cache_.size());
        std::vector<std::pair<int,int>> trimmed;
        trimmed.reserve(hidden_spans_.size());
        for (auto& s : hidden_spans_) {
            int e = std::min(s.second, cache_len);
            if (s.first < e) trimmed.emplace_back(s.first, e);
        }
        hidden_spans_ = std::move(trimmed);
    }

    // Pre-step constraints with the full context sequence
    for (int token : context_cache_) {
        constraints_.step(token, vocab_);
    }

    // For multi-fill: context_cache_ ends with the first FillInStart token, so
    // we're about to sample tokens for block 0. Tell the grammar which track
    // type that block targets so it can enforce the correct melodic/drum rules.
    // (Autoregressive doesn't need this — the agent's Track token is the last
    // Track token in the prompt after the reorder above, so GrammarConstraint's
    // is_drum_ is naturally correct.)
    if (!step_.is_autoregressive && !fillin_drum_order_.empty()) {
        fillin_idx_ = 0;
        constraints_.set_fillin_drum(fillin_drum_order_[0]);
    }
}

bool SessionState::complete() const {
    if (generated_.empty()) return false;
    if (step_.is_autoregressive) {
        return vocab_.is_type(generated_.back(), TokenType::TrackEnd)
            || vocab_.is_type(generated_.back(), TokenType::PieceEnd);
    } else {
        // Infill: complete when we've seen all FILL_IN_END tokens
        int fill_end_count = 0;
        for (int tok : generated_) {
            if (vocab_.is_type(tok, TokenType::FillInEnd)) {
                fill_end_count++;
            }
        }
        return fill_end_count >= static_cast<int>(step_.bars_to_generate.size());
    }
}

std::vector<int> SessionState::context_tokens() const {
    std::vector<int> full_context = context_cache_;
    full_context.insert(full_context.end(), generated_.begin(), generated_.end());
    return full_context;
}

std::vector<bool> SessionState::logit_mask() const {
    std::vector<bool> mask = constraints_.get_mask(vocab_);
    // Invert: ConstraintGraph uses true=disallowed, model needs true=allowed
    for (size_t i = 0; i < mask.size(); ++i) {
        mask[i] = !mask[i];
    }
    return mask;
}

void SessionState::advance(int token) {
    generated_.push_back(token);
    constraints_.step(token, vocab_);

    // When a new FillInStart is generated, advance to the next fill block's
    // drum flag so the grammar knows whether NoteOnset must be followed by
    // NoteDuration (melodic) or may go straight to the next event (drum).
    if (!step_.is_autoregressive
        && vocab_.is_type(token, TokenType::FillInStart)
        && !fillin_drum_order_.empty()) {
        fillin_idx_++;
        if (fillin_idx_ < fillin_drum_order_.size()) {
            constraints_.set_fillin_drum(fillin_drum_order_[fillin_idx_]);
        }
    }
}

Score SessionState::result() const {
    Score out = decoder_.decode(context_tokens());
    // Autoregressive reorder: agent track was moved to the last position
    // before encoding; restore it to ar_original_agent_idx_ so the caller
    // sees its tracks in the original order.
    if (ar_original_agent_idx_ >= 0
        && !out.tracks.empty()
        && ar_original_agent_idx_ < static_cast<int>(out.tracks.size())) {
        auto agent = std::move(out.tracks.back());
        out.tracks.pop_back();
        out.tracks.insert(out.tracks.begin() + ar_original_agent_idx_,
                          std::move(agent));
    }
    // Splice the bars saved before the window-shift back around the decoded
    // window so callers see absolute bar indices over the full piece. The
    // saved bars' note_indices point into original_notes_, NOT out.notes, so
    // we copy the referenced notes into out.notes and remap indices.
    auto splice_one = [&](Bar bar) -> Bar {
        std::vector<int> remapped;
        remapped.reserve(bar.note_indices.size());
        for (int old_idx : bar.note_indices) {
            if (old_idx < 0
                || old_idx >= static_cast<int>(original_notes_.size())) {
                continue;
            }
            int new_idx = static_cast<int>(out.notes.size());
            out.notes.push_back(original_notes_[old_idx]);
            remapped.push_back(new_idx);
        }
        bar.note_indices = std::move(remapped);
        return bar;
    };
    for (size_t ti = 0; ti < out.tracks.size(); ++ti) {
        auto& bars = out.tracks[ti].bars;
        if (ti < prefix_bars_.size() && !prefix_bars_[ti].empty()) {
            std::vector<Bar> remapped;
            remapped.reserve(prefix_bars_[ti].size());
            for (const auto& b : prefix_bars_[ti]) remapped.push_back(splice_one(b));
            bars.insert(bars.begin(), remapped.begin(), remapped.end());
        }
        if (ti < suffix_bars_.size() && !suffix_bars_[ti].empty()) {
            for (const auto& b : suffix_bars_[ti]) bars.push_back(splice_one(b));
        }
    }
    return out;
}

} // namespace midigpt::sampling
