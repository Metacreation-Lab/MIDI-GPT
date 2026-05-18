#include "session_state.h"
#include "../core/logging.h"

namespace midigpt::sampling {

SessionState::SessionState(
    Score                                     context,
    const GenerationStep&                     step,
    const tokenizer::Vocabulary&              vocab,
    const masking::ConstraintGraph&           constraints,
    const tokenizer::Encoder&                 encoder,
    const tokenizer::Decoder&                 decoder
) : context_(std::move(context)), step_(step), vocab_(vocab),
    constraints_(constraints), encoder_(encoder), decoder_(decoder)
{
    // Trim bars to the step's window [start_bar, end_bar)
    // Keep bars [0, end_bar) so bar indices stay valid
    for (auto& track : context_.tracks) {
        if (static_cast<int>(track.bars.size()) > step_.end_bar) {
            track.bars.resize(step_.end_bar);
        }
    }

    tokenizer::EncodeOptions encode_opts;
    encode_opts.window_bars = step_.end_bar - step_.start_bar;

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
    context_cache_ = encoder_.encode(context_, encode_opts);

    if (step_.is_autoregressive) {
        // Strip trailing structural tokens so the model can continue generating
        while (!context_cache_.empty()) {
            TokenType type = vocab_.get_type(context_cache_.back());
            if (type == TokenType::PieceEnd || type == TokenType::TrackEnd ||
                type == TokenType::Bar || type == TokenType::BarEnd ||
                type == TokenType::TimeSig || type == TokenType::NumBars ||
                type == TokenType::MaskBar) {
                context_cache_.pop_back();
            } else {
                break;
            }
        }
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
    return out;
}

} // namespace midigpt::sampling
