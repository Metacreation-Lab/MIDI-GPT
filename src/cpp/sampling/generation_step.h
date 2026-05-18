#pragma once

#include <vector>
#include <set>
#include <tuple>

namespace midigpt::sampling {

struct GenerationStep {
    int start_bar;                                // window start (global bar index)
    int end_bar;                                  // window end (exclusive, global bar index)
    bool is_autoregressive = true;                // true = AR, false = infill
    std::vector<int>                          track_indices;    // global track indices in window
    std::set<std::pair<int,int>>              bars_to_generate; // (global_track, global_bar)
    std::vector<std::tuple<int,int,int,int>>  bar_mapping;      // (local_track, local_bar, global_track, global_bar)
    std::vector<std::vector<bool>>            context;          // [num_tracks][num_bars] — true = context bar
};

} // namespace midigpt::sampling
