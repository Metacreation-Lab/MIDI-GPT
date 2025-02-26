#include "train_config.h"


namespace data_structures {

    TrainConfig::TrainConfig() {
        num_bars = 4;
        min_tracks = 1;
        max_tracks = 12;
        max_mask_percentage = 0.75;
        use_microtiming = false;
        microtiming = 0.9;
        no_max_length = false;
        resolution = 12;
        delta_resolution = 1920;
        decode_resolution = delta_resolution;
    }

    std::map<std::string, std::string> TrainConfig::ToJson() {
        std::map<std::string, std::string> json_config;
        json_config["num_bars"] = std::to_string(num_bars);
        json_config["min_tracks"] = std::to_string(min_tracks);
        json_config["max_tracks"] = std::to_string(max_tracks);
        json_config["max_mask_percentage"] = std::to_string(max_mask_percentage);
        json_config["use_microtiming"] = std::to_string((int)use_microtiming);
        json_config["microtiming"] = std::to_string(microtiming);
        json_config["no_max_length"] = std::to_string((int)no_max_length);
        json_config["resolution"] = std::to_string(resolution);
        json_config["decode_resolution"] = std::to_string(decode_resolution);
        json_config["delta_resolution"] = std::to_string(delta_resolution);
        return json_config;
    }

    void TrainConfig::FromJson(std::map<std::string, std::string>& json_config) {
        num_bars = stoi(json_config["num_bars"]);
        min_tracks = stoi(json_config["min_tracks"]);
        max_tracks = stoi(json_config["max_tracks"]);
        max_mask_percentage = stof(json_config["max_mask_percentage"]);
        microtiming = stoi(json_config["microtiming"]);
        use_microtiming = (bool)stoi(json_config["use_microtiming"]);
        no_max_length = (bool)stoi(json_config["no_max_length"]);
        resolution = stoi(json_config["resolution"]);
        decode_resolution = stoi(json_config["decode_resolution"]);
        delta_resolution = stoi(json_config["delta_resolution"]);
    }
}