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

    TrainConfig();

    std::map<std::string, std::string> ToJson();
    void FromJson(std::map<std::string, std::string>& json_config);
};

}
// END OF NAMESPACE