#pragma once

#include <vector>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <stdexcept>
#include <algorithm>
#include <nlohmann/json.hpp>

namespace midigpt::tokenizer {

// ============================================================
// TimeSignatureList
// An ordered list of supported time signatures.
// Token value = index in the list. Encode/decode by lookup.
// ============================================================
class TimeSignatureList {
public:
    TimeSignatureList() = default;

    explicit TimeSignatureList(std::vector<std::pair<int,int>> signatures)
        : signatures_(std::move(signatures))
    {
        for (int i = 0; i < static_cast<int>(signatures_.size()); ++i) {
            lookup_[signatures_[i]] = i;
        }
    }

    int encode(int numerator, int denominator) const {
        auto it = lookup_.find({numerator, denominator});
        if (it == lookup_.end()) {
            throw std::runtime_error(
                "Unknown time signature: " + std::to_string(numerator) +
                "/" + std::to_string(denominator));
        }
        return it->second;
    }

    std::pair<int,int> decode(int index) const {
        if (index < 0 || index >= static_cast<int>(signatures_.size())) {
            throw std::runtime_error("Time signature index out of range: " +
                                     std::to_string(index));
        }
        return signatures_[index];
    }

    int size() const { return static_cast<int>(signatures_.size()); }

    static TimeSignatureList from_json(const nlohmann::json& j) {
        std::vector<std::pair<int,int>> sigs;
        for (const auto& s : j) {
            std::string str = s.get<std::string>();
            auto slash = str.find('/');
            if (slash == std::string::npos) {
                throw std::runtime_error("Invalid time signature format: " + str);
            }
            int num = std::stoi(str.substr(0, slash));
            int den = std::stoi(str.substr(slash + 1));
            sigs.push_back({num, den});
        }
        return TimeSignatureList(std::move(sigs));
    }

    nlohmann::json to_json() const {
        nlohmann::json arr = nlohmann::json::array();
        for (const auto& [num, den] : signatures_) {
            arr.push_back(std::to_string(num) + "/" + std::to_string(den));
        }
        return arr;
    }

private:
    std::vector<std::pair<int,int>> signatures_;
    std::map<std::pair<int,int>, int> lookup_;
};


// ============================================================
// VelocityQuantizer
// Uniform quantization: 128 MIDI velocities → N levels.
// Fully derived from num_levels (= domain_size in config).
//
//   encode: v == 0 ? 0 : min(1 + v*(N-1)/128, N-1)
//   decode: second element of the bin (bin[1])
// ============================================================
class VelocityQuantizer {
public:
    explicit VelocityQuantizer(int num_levels) : num_levels_(num_levels) {
        if (num_levels_ < 2) {
            throw std::runtime_error("VelocityQuantizer requires at least 2 levels");
        }
        // Pre-compute the decode table: pick the MAX MIDI velocity in each bin.
        // Matches original enums::DEFAULT_VELOCITY_MAP inverse (each level maps
        // back to the highest MIDI velocity that quantized into it).
        decode_table_.assign(num_levels_, 0);
        for (int v = 1; v < 128; ++v) {
            int level = encode(v);
            decode_table_[level] = v; // overwrite — final iteration wins (max)
        }
    }

    int encode(int velocity) const {
        if (velocity <= 0) return 0;
        if (velocity >= 127) return num_levels_ - 1;
        return std::min(1 + velocity * (num_levels_ - 1) / 128, num_levels_ - 1);
    }

    int decode(int level) const {
        if (level < 0 || level >= num_levels_) {
            throw std::runtime_error("Velocity level out of range: " +
                                     std::to_string(level));
        }
        return decode_table_[level];
    }

    int num_levels() const { return num_levels_; }

private:
    int num_levels_;
    std::vector<int> decode_table_;
};


// ============================================================
// InstrumentGrouping
// Merge groups: specifies which GM instruments are equivalent.
// Dense reindexing is derived automatically.
//
// Config format: list of lists, e.g. [[0,1,2], [4,5], [88..95]]
// Unmentioned instruments map to themselves (1:1).
// Dense IDs are assigned in ascending order of representative.
// ============================================================
class InstrumentGrouping {
public:
    InstrumentGrouping() = default;

    explicit InstrumentGrouping(std::vector<std::vector<int>> merge_groups,
                                int total_instruments = 128)
        : merge_groups_(std::move(merge_groups))
    {
        // Step 1: Map each instrument to its representative (first in group)
        std::map<int, int> representative;
        for (int i = 0; i < total_instruments; ++i) {
            representative[i] = i;
        }
        for (const auto& group : merge_groups_) {
            if (group.empty()) continue;
            int rep = group[0];
            for (int inst : group) {
                representative[inst] = rep;
            }
        }

        // Step 2: Collect unique representatives in sorted order
        std::vector<int> unique_reps;
        {
            std::set<int> seen;
            for (int i = 0; i < total_instruments; ++i) {
                int rep = representative[i];
                if (seen.insert(rep).second) {
                    unique_reps.push_back(rep);
                }
            }
        }
        std::sort(unique_reps.begin(), unique_reps.end());

        // Step 3: Assign dense IDs
        std::map<int, int> dense_id;
        for (int idx = 0; idx < static_cast<int>(unique_reps.size()); ++idx) {
            dense_id[unique_reps[idx]] = idx;
        }

        // Step 4: Build forward (instrument → dense ID) and reverse maps
        num_groups_ = static_cast<int>(unique_reps.size());
        for (int i = 0; i < total_instruments; ++i) {
            forward_[i] = dense_id[representative[i]];
        }
        for (int idx = 0; idx < static_cast<int>(unique_reps.size()); ++idx) {
            reverse_[idx] = unique_reps[idx];
        }
    }

    int encode(int midi_instrument) const {
        auto it = forward_.find(midi_instrument);
        if (it != forward_.end()) return it->second;
        return 0; // fallback for out-of-range instruments
    }

    int decode(int group_id) const {
        auto it = reverse_.find(group_id);
        if (it != reverse_.end()) return it->second;
        return 0;
    }

    int num_groups() const { return num_groups_; }

    static InstrumentGrouping from_json(const nlohmann::json& j,
                                        int total_instruments = 128) {
        std::vector<std::vector<int>> groups;
        for (const auto& arr : j) {
            std::vector<int> group;
            for (const auto& val : arr) {
                group.push_back(val.get<int>());
            }
            groups.push_back(std::move(group));
        }
        return InstrumentGrouping(std::move(groups), total_instruments);
    }

    nlohmann::json to_json() const {
        nlohmann::json arr = nlohmann::json::array();
        for (const auto& group : merge_groups_) {
            arr.push_back(group);
        }
        return arr;
    }

private:
    std::vector<std::vector<int>> merge_groups_;
    std::map<int, int> forward_;   // midi_instrument → dense group ID
    std::map<int, int> reverse_;   // dense group ID → representative instrument
    int num_groups_ = 0;
};

/**
 * Maps arbitrary integer values to a dense [0, N) range and back.
 * Used for domains like NumBars {4, 8, 12, 16} or TrackType {10, 11}.
 */
class ValueMapper {
public:
    ValueMapper() = default;
    ValueMapper(const std::vector<int>& values) : values_(values) {
        for (size_t i = 0; i < values_.size(); ++i) {
            to_index_[values_[i]] = static_cast<int>(i);
        }
    }

    int encode(int value) const {
        auto it = to_index_.find(value);
        if (it != to_index_.end()) return it->second;

        std::string allowed;
        for (size_t i = 0; i < values_.size(); ++i) {
            if (i) allowed += ", ";
            allowed += std::to_string(values_[i]);
        }
        throw std::invalid_argument(
            "ValueMapper::encode: value " + std::to_string(value) +
            " not in vocab domain [" + allowed + "]");
    }

    bool contains(int value) const {
        return to_index_.find(value) != to_index_.end();
    }

    int decode(int index) const {
        if (index < 0 || index >= static_cast<int>(values_.size())) {
            throw std::runtime_error("Index out of range for value map");
        }
        return values_[index];
    }

    size_t size() const { return values_.size(); }

    const std::vector<int>& values() const { return values_; }

private:
    std::vector<int> values_;
    std::map<int, int> to_index_;
};

} // namespace midigpt::tokenizer
