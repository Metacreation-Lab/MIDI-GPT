#pragma once

#include <map>
#include <string>

// START OF NAMESPACE
namespace data_structures {

class TrainConfig {
public:
    int num_bars;
    int min_tracks;
    int max_tracks;
    float max_mask_percentage;
    bool use_microtiming;
    float microtiming;
    bool no_max_length;
    int resolution;
    int decode_resolution;
    int delta_resolution;

    // Mask-bar augmentation (forwarded to EncoderConfig at read_batch time)
    bool do_mask_augmentation;
    float mask_apply_probability;  // gate: fraction of samples that get any masking
    int mask_type;                 // 0=random, 1=structured-future, 2=mixed
    float mask_bar_fraction;       // max fraction of bars masked when gate fires
    int mask_max_lookahead;

    TrainConfig();

    std::map<std::string, std::string> ToJson();
    void FromJson(std::map<std::string, std::string>& json_config);
};

}
// END OF NAMESPACE