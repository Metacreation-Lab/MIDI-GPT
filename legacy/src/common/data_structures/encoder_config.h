#pragma once

#include <vector>
#include <tuple>
#include <set>
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
            force_instrument = false;
            use_note_duration_encoding = false;
            use_absolute_time_encoding = false;
            mark_time_sigs = false;
            mark_note_duration_quantile = false;
            mark_polyphony_quantile = false;
            mark_drum_density = false;
            use_drum_offsets = false;
            min_tracks = 1;
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
        bool force_instrument;
        bool use_note_duration_encoding;
        bool use_absolute_time_encoding;
        bool mark_time_sigs;
        bool mark_note_duration_quantile;
        bool mark_polyphony_quantile;
        bool mark_drum_density;
        bool use_drum_offsets;
        int min_tracks;
        std::set<std::tuple<int, int>> multi_fill;

        // -------------------------------------------------------
        // Mask-bar augmentation (applied on-the-fly during encode)
        // -------------------------------------------------------
        // Whether to apply stochastic mask-bar augmentation for training.
        bool do_mask_augmentation = false;
        // Gate: probability of applying any masking on a given sample.
        // When the gate does not fire, the sample is encoded normally (no mask tokens).
        // Analogous to the 0.75 gate used for bar infilling.
        float mask_apply_probability = 0.5f;
        // 0 = random (sample a random number of bars up to mask_bar_fraction and mask them)
        // 1 = structured-future (mask a contiguous suffix of k bars, k ~ Uniform[1, mask_max_lookahead])
        // 2 = mixed  (50% random, 50% structured-future)
        int mask_type = 0;
        // Maximum fraction of bars to mask when the random mode fires.
        // Actual count is sampled uniformly from [1, max(1, floor(mask_bar_fraction * num_bars))].
        float mask_bar_fraction = 0.5f;
        // Maximum lookahead depth for structured masking (mode 1 and 2).
        int mask_max_lookahead = 4;
        // RNG seed for mask augmentation. -1 = new random seed each call.
        int mask_seed = -1;

        // -------------------------------------------------------
        // Partial-track encoding (used for suffix-autoregressive prompt)
        // -------------------------------------------------------
        // If >= 0, encode only partial_encode_track_bars bars for this track index
        // and omit the TRACK_END token (the model continues generating from there).
        int partial_encode_track_index = -1;
        int partial_encode_track_bars  = -1;

        // -------------------------------------------------------
        // Runtime state — populated by apply_mask_augmentation().
        // NOT serialized; cleared at the start of each encode_piece call.
        // -------------------------------------------------------
        // (track_index, bar_index) pairs that should be emitted as BAR MASK_BAR BAR_END.
        std::set<std::tuple<int, int>> mask_bars;
    };
}