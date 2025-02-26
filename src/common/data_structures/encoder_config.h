#pragma once
  
#include <vector>
#include <tuple>
#include <map>
#include <random>

namespace data_structures {
    class EncoderConfig {
    public:
        EncoderConfig() {
            both_in_one = false;
            unquantized = false;
            do_multi_fill = false;
            use_velocity_levels = false;
            use_microtiming = false;
            transpose = 0;
            resolution = 12;
            decode_resolution = resolution;
            decode_final = false;
            delta_resolution = 1920;
        }

        std::map<std::string, std::string> ToJson() {
            std::map<std::string, std::string> json_config;

            json_config["both_in_one"] = std::to_string((int)both_in_one);
            json_config["unquantized"] = std::to_string((int)unquantized);
            json_config["do_multi_fill"] = std::to_string((int)do_multi_fill);
            json_config["use_velocity_levels"] = std::to_string((int)use_velocity_levels);
            json_config["use_microtiming"] = std::to_string((int)use_microtiming);
            json_config["transpose"] = std::to_string(transpose);
            json_config["resolution"] = std::to_string(resolution);
            json_config["decode_resolution"] = std::to_string(decode_resolution);
            json_config["decode_final"] = std::to_string((int)decode_final);
            json_config["delta_resolution"] = std::to_string(delta_resolution);
            return json_config;
        }

        void FromJson(const std::map<std::string, std::string>& json_config) {
            try {
                both_in_one = (bool)std::stoi(json_config.at("both_in_one"));
                unquantized = (bool)std::stoi(json_config.at("unquantized"));
                do_multi_fill = (bool)std::stoi(json_config.at("do_multi_fill"));
                use_velocity_levels = (bool)std::stoi(json_config.at("use_velocity_levels"));
                use_microtiming = (bool)std::stoi(json_config.at("use_microtiming"));
                transpose = std::stoi(json_config.at("transpose"));
                resolution = std::stoi(json_config.at("resolution"));
                decode_resolution = std::stoi(json_config.at("decode_resolution"));
                decode_final = (bool)std::stoi(json_config.at("decode_final"));
                delta_resolution = std::stoi(json_config.at("delta_resolution"));
            } catch (const std::out_of_range& e) {
                throw std::invalid_argument("Missing required key in JSON config: " + std::string(e.what()));
            } catch (const std::invalid_argument& e) {
                throw std::invalid_argument("Invalid value type in JSON config: " + std::string(e.what()));
            }
        }

        int delta_to_step(int delta, int res) {
            if (!use_microtiming) {
                return 0;
            } else { 
                return (int)(delta * res / delta_resolution);
            }
        }

        int step_to_delta(float step, int res) {
            if (!use_microtiming) {
                return 0;
            } else { 
                return round(delta_resolution * step / res);
            }
        }

        int step_to_delta(int step, int res) {
            if (!use_microtiming) {
                return 0;
            } else { 
                return round(delta_resolution * step / res);
            }
        }

        bool both_in_one;
        bool unquantized;
        bool do_multi_fill;
        bool use_velocity_levels;
        bool use_microtiming;
        int transpose;
        int resolution;
        int decode_resolution;
        bool decode_final;
        int delta_resolution;
        std::set<std::tuple<int, int>> multi_fill;
    };
}