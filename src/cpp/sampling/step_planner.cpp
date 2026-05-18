#include "step_planner.h"
#include <algorithm>
#include <set>

namespace midigpt::sampling {

StepPlanner::StepPlanner(const SelectionMask& mask, const tokenizer::EncoderConfig& config,
                         int bars_per_step, int tracks_per_step)
    : mask_(mask), config_(config),
      bars_per_step_(bars_per_step > 0 ? bars_per_step : config.model_dim),
      tracks_per_step_(tracks_per_step) {}

std::vector<GenerationStep> StepPlanner::plan() const {
    if (mask_.selected.empty()) return {};
    int nt = static_cast<int>(mask_.selected.size());
    int nb = static_cast<int>(mask_.selected[0].size());

    std::vector<GenerationStep> steps;
    std::vector<std::vector<bool>> generated(nt, std::vector<bool>(nb, false));

    // First pass: autoregressive steps (tracks with autoregressive=true)
    find_steps_inner(steps, generated, true);
    // Second pass: infill steps (tracks with autoregressive=false)
    find_steps_inner(steps, generated, false);
    return steps;
}

void StepPlanner::find_steps_inner(std::vector<GenerationStep>& steps,
                                    std::vector<std::vector<bool>>& generated,
                                    bool autoregressive) const {
    int nt = static_cast<int>(mask_.selected.size());
    if (nt == 0) return;
    int nb = static_cast<int>(mask_.selected[0].size());
    int model_dim = config_.model_dim;
    int bps = std::clamp(bars_per_step_, 1, model_dim);
    int tps = std::clamp(tracks_per_step_, 1, nt);

    // Build effective selection: selected & (autoregressive XOR infill)
    // For AR:    sel = selected & autoregressive_mask
    // For infill: sel = selected & ~autoregressive_mask
    std::vector<std::vector<bool>> sel(nt, std::vector<bool>(nb, false));
    for (int i = 0; i < nt; ++i) {
        bool is_ar = (i < static_cast<int>(mask_.autoregressive.size()) && mask_.autoregressive[i]);
        bool is_ignored = (i < static_cast<int>(mask_.ignore.size()) && mask_.ignore[i]);
        if (is_ignored) continue;
        bool include = autoregressive ? is_ar : !is_ar;
        if (!include) continue;
        for (int j = 0; j < nb; ++j) {
            sel[i][j] = mask_.selected[i][j];
        }
    }

    // Context amount: AR gets all context before, infill centers
    int num_context = autoregressive ? model_dim - bps : (model_dim - bps) / 2;

    // Iterate in (tracks_per_step, bars_per_step) grid
    for (int i = 0; i < nt; i += tps) {
        for (int j = 0; j < nb; j += bps) {
            int num_tracks = std::min(tps, nt - i);

            // --- Determine window start t ---
            int t;
            // Kernel: marks which bars within the model_dim window are generation targets
            std::vector<std::vector<bool>> kernel(num_tracks, std::vector<bool>(model_dim, false));

            if (autoregressive) {
                // Position window so generation bars are right-aligned (maximize past context)
                int target_end_bar = j + bps - 1;
                t = std::clamp(target_end_bar - model_dim + 1, 0, std::max(0, nb - model_dim));
                int local_start = j - t;
                int local_end = std::min(model_dim, local_start + bps);
                for (int ti = 0; ti < num_tracks; ++ti) {
                    for (int bi = local_start; bi < local_end; ++bi) {
                        kernel[ti][bi] = true;
                    }
                }
            } else {
                // Center generation bars in window
                t = std::clamp(j - num_context, 0, std::max(0, nb - model_dim));
                int local_start = j - t;
                int local_end = std::min(model_dim, local_start + bps);
                for (int ti = 0; ti < num_tracks; ++ti) {
                    for (int bi = local_start; bi < local_end; ++bi) {
                        kernel[ti][bi] = true;
                    }
                }
            }

            int window_end = t + model_dim;
            int a = i + num_tracks; // track range end

            // --- Build step matrix: sel * kernel, minus already generated ---
            std::vector<std::vector<bool>> step_matrix(nt, std::vector<bool>(nb, false));
            for (int ti = i; ti < a; ++ti) {
                for (int bi = t; bi < std::min(window_end, nb); ++bi) {
                    int ki = ti - i;
                    int kj = bi - t;
                    if (kj < model_dim && sel[ti][bi] && kernel[ki][kj]) {
                        step_matrix[ti][bi] = true;
                    }
                }
            }
            // For AR: remove already generated bars
            if (autoregressive) {
                for (int ti = i; ti < a; ++ti) {
                    for (int bi = t; bi < std::min(window_end, nb); ++bi) {
                        if (generated[ti][bi]) {
                            step_matrix[ti][bi] = false;
                        }
                    }
                }
            }

            // --- Build context matrix ---
            std::vector<std::vector<bool>> ctx(nt, std::vector<bool>(nb, false));
            for (int ti = 0; ti < nt; ++ti) {
                for (int bi = t; bi < std::min(window_end, nb); ++bi) {
                    bool is_ignored = (ti < static_cast<int>(mask_.ignore.size()) && mask_.ignore[ti]);
                    ctx[ti][bi] = !is_ignored && !step_matrix[ti][bi];
                }
            }
            if (autoregressive) {
                // For AR tracks: context = previously generated bars only
                // For non-selected tracks: context stays as computed
                // h = "does this track have any selection?" (max along bars axis)
                for (int ti = 0; ti < nt; ++ti) {
                    bool has_selection = false;
                    for (int bi = 0; bi < nb; ++bi) {
                        if (sel[ti][bi]) { has_selection = true; break; }
                    }
                    for (int bi = t; bi < std::min(window_end, nb); ++bi) {
                        if (has_selection) {
                            ctx[ti][bi] = generated[ti][bi];
                        }
                        // else: ctx stays as computed (not ignored, not step)
                    }
                }
            }

            // Check if any bars to generate in this step
            bool has_gen = false;
            for (int ti = 0; ti < nt; ++ti) {
                for (int bi = 0; bi < nb; ++bi) {
                    if (step_matrix[ti][bi]) { has_gen = true; break; }
                }
                if (has_gen) break;
            }

            if (has_gen) {
                GenerationStep gs;
                gs.start_bar = t;
                gs.end_bar = std::min(window_end, nb);
                gs.is_autoregressive = autoregressive;
                gs.context = ctx;

                // Collect track indices (tracks that have step or context in window)
                std::set<int> track_set;
                for (int ti = 0; ti < nt; ++ti) {
                    bool track_used = false;
                    for (int bi = t; bi < gs.end_bar; ++bi) {
                        if (step_matrix[ti][bi] || ctx[ti][bi]) {
                            track_used = true;
                        }
                        if (step_matrix[ti][bi]) {
                            gs.bars_to_generate.insert({ti, bi});
                        }
                    }
                    if (track_used) {
                        track_set.insert(ti);
                    }
                }
                gs.track_indices = std::vector<int>(track_set.begin(), track_set.end());

                // Build bar_mapping: (local_track_in_subset, local_bar, global_track, global_bar)
                // local_track = index within track_indices
                for (const auto& [gt, gb] : gs.bars_to_generate) {
                    auto it = std::find(gs.track_indices.begin(), gs.track_indices.end(), gt);
                    int local_track = static_cast<int>(std::distance(gs.track_indices.begin(), it));
                    int local_bar = gb - gs.start_bar;
                    gs.bar_mapping.push_back({local_track, local_bar, gt, gb});
                }

                steps.push_back(std::move(gs));
            }

            // Update generated matrix
            for (int ti = i; ti < a; ++ti) {
                for (int bi = t; bi < std::min(window_end, nb); ++bi) {
                    if (step_matrix[ti][bi]) {
                        generated[ti][bi] = true;
                    }
                }
            }
        }
    }
}

} // namespace midigpt::sampling
