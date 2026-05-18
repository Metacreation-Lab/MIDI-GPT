#pragma once

#include <vector>

namespace midigpt::sampling {

struct SelectionMask {
    std::vector<std::vector<bool>> selected;   // [track][bar] — true = generate here
    std::vector<bool> autoregressive;          // per track
    std::vector<bool> ignore;                  // per track — excluded from context
};

} // namespace midigpt::sampling
