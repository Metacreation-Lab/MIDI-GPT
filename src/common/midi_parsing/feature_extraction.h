#pragma once

#include <string>
#include <vector>

#include "../../common/midi_parsing/util_protobuf.h"
#include "../../../libraries/protobuf/build/midi.pb.h"

namespace feature_extraction {

class FEATURE_EXTRACTOR {
public:

    virtual ~FEATURE_EXTRACTOR() {}

    virtual void compute_track_feature(midi::Piece *x, int track_num, midi::Features *f, std::string &filepath) {
        throw std::runtime_error("FEATURE_EXTRACTOR::compute_track_feature() not implemented");
    }

    virtual void compute_piece_feature(midi::Piece *x, midi::Features *f, std::string &filepath) {
        throw std::runtime_error("FEATURE_EXTRACTOR::compute_piece_feature() not implemented");
    }

    void compute_feature(midi::Piece *x, midi::Features *f, std::string &filepath) {
        if (track_level) {
            for (int track_num=0; track_num<x->tracks_size(); track_num++) {
                compute_track_feature(x, track_num, f, filepath);
            }
        }
        else {
            compute_piece_feature(x, f, filepath);
        }
    }

    bool quantized;
    bool track_level;

};

class PitchRangeFeature : public FEATURE_EXTRACTOR {
public:

    PitchRangeFeature() {
        quantized = false;
        track_level = true;
    }

    void compute_track_feature(midi::Piece *x, int track_num, midi::Features *f, std::string &filepath) {
        const auto track = x->tracks(track_num);

        // ignore drum tracks
        if (data_structures::is_drum_track(track.track_type())) {
            return;
        }

        int min_pitch = INT_MAX;
        int max_pitch = INT_MIN;
        for (const auto &bar : track.bars()) {
            for (const auto &event_index : bar.events()) {
                if (x->events(event_index).velocity()) {
                    if (x->events(event_index).pitch() < min_pitch) {
                        min_pitch = x->events(event_index).pitch();
                    }
                    if (x->events(event_index).pitch() > max_pitch) {
                        max_pitch = x->events(event_index).pitch();
                    }
                }
            }
        }

        auto fm = f->add_pitch_range();
        fm->set_instrument(track.instrument());
        fm->set_min(min_pitch);
        fm->set_max(max_pitch);
    }
};

class DownbeatProportionFeature : public FEATURE_EXTRACTOR {
public:

    DownbeatProportionFeature() {
        quantized = true;
        track_level = true;
    }

    void compute_track_feature(midi::Piece *x, int track_num, midi::Features *f, std::string &filepath) {
        const auto track = x->tracks(track_num);

        if (!x->internal_has_time_signatures()) {
            return;
        }

        int downbeat_count = 0;
        int onset_count = 0;

        std::map<int,int> onset_counts;
        for (const auto &bar : track.bars()) {
            for (const auto &event_index : bar.events()) {
                if (x->events(event_index).velocity()) {

                    onset_counts[x->events(event_index).time()] += 1;
                    
                    if (x->events(event_index).time() == 0) {
                        downbeat_count++;
                    }
                    onset_count++;

                }
            }
        }

        int max_non_downbeat = 1;
        for (const auto &kv : onset_counts) {
            if (kv.first != 0) {
                max_non_downbeat = std::max(max_non_downbeat, kv.second);
            }
        }

        auto fm = f->add_downbeat_proportion();
        fm->set_instrument(track.instrument());
        fm->set_is_drum(data_structures::is_drum_track(track.track_type()));
        fm->set_filepath(filepath);
        fm->set_track_num(track_num);

        fm->set_downbeat_proportion(float(onset_counts[0]) / float(max_non_downbeat));

    }
};

std::map<int,int> compute_metric_depth_counts(midi::Piece *x, int track_num, int max_depth, int offset) {
    const auto track = x->tracks(track_num);

    int max_duple_depth = 0;
    int max_triplet_depth = 0;
    int total_depth = max_depth * 2;

    while ((x->internal_ticks_per_quarter() % int(pow(2,max_duple_depth))) == 0) {
        max_duple_depth += 1;
    }
    while ((x->internal_ticks_per_quarter() * 2) % (int(pow(2,max_triplet_depth)) * 3) == 0) {
        max_triplet_depth += 1;
    }

    max_duple_depth = std::min(max_duple_depth, max_depth);
    max_triplet_depth = std::min(max_triplet_depth, max_depth);

    std::map<int,int> metric_depth_counts;
    for (const auto &bar : track.bars()) {
        for (const auto &event_index : bar.events()) {
            if (x->events(event_index).velocity()) {
                bool found_depth = false;
                for (int i=0; i<max_duple_depth; i++) {
                    int period = (x->internal_ticks_per_quarter()) / int(pow(2,i));
                    if (((x->events(event_index).time()) + offset) % period == 0) {
                        metric_depth_counts[2*i] += 1;
                        found_depth = true;
                        break;
                    }
                }
                if (!found_depth) {
                    for (int i=0; i<max_triplet_depth; i++) {
                        int period = (x->internal_ticks_per_quarter() * 2) / (int(pow(2,i)) * 3);
                        if ((x->events(event_index).time() + offset) % period == 0) {
                            metric_depth_counts[2*i + 1] += 1;
                            found_depth = true;
                            break;
                        }
                    }
                }
                if (!found_depth) {
                    metric_depth_counts[total_depth] += 1;
                }
            }
        }
    }
    return metric_depth_counts;
}

class MetricDepthFeature : public FEATURE_EXTRACTOR {
public:

    MetricDepthFeature() {
        quantized = false;
        track_level = true;
    }

    void compute_track_feature(midi::Piece *x, int track_num, midi::Features *f, std::string &filepath) {
        
        int max_depth = 6;
        auto metric_depth_counts = compute_metric_depth_counts(x, track_num, max_depth, 0);
        const auto track = x->tracks(track_num);

        auto fm = f->add_metric_depth();
        fm->set_filepath(filepath);
        fm->set_track_num(track_num);
        fm->set_instrument(track.instrument());
        fm->set_is_drum(data_structures::is_drum_track(track.track_type()));
        fm->set_has_time_signatures(x->internal_has_time_signatures());
        fm->set_tpq(x->internal_ticks_per_quarter());
        for (int i=0; i<=max_depth*2; i++) {
            if (metric_depth_counts.find(i) == metric_depth_counts.end()) {
                fm->add_metric_depth(0);
            } else {
                fm->add_metric_depth(metric_depth_counts[i]);
            }
        }
    }
};

class MostFrequentMetricDepthFeature : public FEATURE_EXTRACTOR {
public:

    MostFrequentMetricDepthFeature() {
        quantized = false;
        track_level = true;
    }

    void compute_track_feature(midi::Piece *x, int track_num, midi::Features *f, std::string &filepath) {
        
        int max_depth = 6;
        auto metric_depth_counts = compute_metric_depth_counts(x, track_num, max_depth, 0);
        const auto track = x->tracks(track_num);


        int max_count = 0;
        int max_index = 0;
        for (const auto &kv : metric_depth_counts) {
            if (kv.second > max_count) {
                max_count = kv.second;
                max_index = kv.first;
            }
        }

        auto fm = f->add_most_frequent_metric_depth();
        fm->set_filepath(filepath);
        fm->set_track_num(track_num);
        fm->set_instrument(track.instrument());
        fm->set_is_drum(data_structures::is_drum_track(track.track_type()));
        fm->set_has_time_signatures(x->internal_has_time_signatures());
        fm->set_tpq(x->internal_ticks_per_quarter());
        fm->set_most_frequent_metric_depth(max_index);
    }

};

class MedianMetricDepthFeature : public FEATURE_EXTRACTOR {
public:

    MedianMetricDepthFeature() {
        quantized = false;
        track_level = true;
    }

    void compute_track_feature(midi::Piece *x, int track_num, midi::Features *f, std::string &filepath) {
        
        int max_depth = 6;
        auto metric_depth_counts = compute_metric_depth_counts(x, track_num, max_depth, 0);
        const auto track = x->tracks(track_num);


        int total = 0;
        for (const auto &kv : metric_depth_counts) {
            total += kv.second;
        }
        
        int median_count = total / 2;
        int median_depth = 0;
        int cumulative = 0;
        for (const auto &kv : metric_depth_counts) {
            if ((median_count >= cumulative) && (median_count < cumulative + kv.second)) {
                median_depth = kv.first;
            }
            cumulative += kv.second;
        }


        auto fm = f->add_median_metric_depth();
        fm->set_filepath(filepath);
        fm->set_track_num(track_num);
        fm->set_instrument(track.instrument());
        fm->set_is_drum(data_structures::is_drum_track(track.track_type()));
        fm->set_has_time_signatures(x->internal_has_time_signatures());
        fm->set_tpq(x->internal_ticks_per_quarter());
        fm->set_median_metric_depth(median_depth);
    }
};

class AlignedMetricDepthFeature : public FEATURE_EXTRACTOR {
public:

    AlignedMetricDepthFeature() {
        quantized = true;
        track_level = true;
    }

    void compute_track_feature(midi::Piece *x, int track_num, midi::Features *f, std::string &filepath) {

        x->set_internal_ticks_per_quarter(12);

        int max_depth = 6;
        int tpq = x->internal_ticks_per_quarter();
        const auto track = x->tracks(track_num);

        if (tpq > 10000) {
            throw std::runtime_error("AlignedMetricDepthFeature::compute_track_feature() must have tpq <= 10000.");
        }

        int best_score_offset = 0;
        int best_score = max_depth * 100;
        for (int offset=0; offset<tpq; offset++) {
            auto metric_depth_counts = compute_metric_depth_counts(x, track_num, max_depth, offset);

            // need to get median score
            int total = 0;
            for (const auto &kv : metric_depth_counts) {
                total += kv.second;
            }
            
            int median_score = 0;
            int cumulative = 0;
            for (const auto &kv : metric_depth_counts) {
                if ((total >= cumulative) && (total < cumulative + kv.second)) {
                    median_score = kv.first;
                }
                cumulative += kv.second;
            }
            
            if (median_score < best_score) {
                best_score = median_score;
                best_score_offset = offset;
            }

        }

        if (best_score_offset != 0) {
            std::cout << "FOUND INVALID MIDI FILE :: " << filepath << std::endl;
        }

        auto fm = f->add_aligned_metric_depth();
        fm->set_filepath(filepath);
        fm->set_track_num(track_num);
        fm->set_instrument(track.instrument());
        fm->set_is_drum(data_structures::is_drum_track(track.track_type()));
        
        fm->set_aligned_offset(best_score_offset);
    }

};

class SimultaneousOnsetFeature : public FEATURE_EXTRACTOR {
public:

    SimultaneousOnsetFeature() {
        quantized = false;
        track_level = true;
    }

    void compute_track_feature(midi::Piece *x, int track_num, midi::Features *f, std::string &filepath) {

        const auto track = x->tracks(track_num);

        int simultaneous_onset_count = 0;
        for (const auto &bar : track.bars()) {
            std::map<int,int> onsets;
            for (const auto &event_index : bar.events()) {
                if (x->events(event_index).velocity()) {
                    onsets[x->events(event_index).time()] += 1;
                }
            }
            for (const auto &kv : onsets) {
                simultaneous_onset_count += (int)(kv.second > 1);
            }
        }

        auto fm = f->add_simultaneous_onset();
        fm->set_filepath(filepath);
        fm->set_track_num(track_num);
        fm->set_instrument(track.instrument());
        fm->set_is_drum(data_structures::is_drum_track(track.track_type()));
        
        fm->set_simultaneous_onset_count(simultaneous_onset_count);
    }

};

template <typename T>
double standardDeviation(std::vector<T> &xs) {
    double total = 0;
    double stdev = 0;
    for (const auto &x : xs) {
        total += x;
    }
    double mean = total / xs.size();
    for (const auto &x : xs) {
        stdev += pow(x - mean, 2);
    }
    return sqrt(stdev / xs.size());
}

template <typename T>
T median(std::vector<T> &xs) {
    std::sort(xs.begin(), xs.end());
    return xs[xs.size() / 2];
}

// basic feature to measure drum presence
class DrumPresenceFeature : public FEATURE_EXTRACTOR {
public:

    DrumPresenceFeature() {
        quantized = true;
        track_level = false;
    }

    void compute_piece_feature(midi::Piece *x, midi::Features *f, std::string &filepath) {
        std::map<int,int> drum_counts_per_bar;
        std::map<int,int> inst_counts_per_bar;
        for (const auto &track : x->tracks()) {
            int bar_num = 0;
            bool is_drum = data_structures::is_drum_track(track.track_type());
            for (const auto &bar : track.bars()) {
                for (const auto &event_index : bar.events()) {
                    auto event = x->events(event_index);
                    if (event.velocity()) {
                        if (is_drum) {
                            drum_counts_per_bar[bar_num]++;
                        } else {
                            inst_counts_per_bar[bar_num]++;
                        }
                    }
                }
                bar_num++;
            }
        }

        int count = 0;
        for (const auto &kv : inst_counts_per_bar) {
            count += (int)(drum_counts_per_bar.find(kv.first) != drum_counts_per_bar.end());
        }

        if (inst_counts_per_bar.size() == 0) {
            return; // invalid midi file or only drums
        }

        auto fm = f->add_drum_presence();
        fm->set_filepath(filepath);
        fm->set_drum_presence((double)count / (double)inst_counts_per_bar.size());

    }
};

class BeatStabilityFeature : public FEATURE_EXTRACTOR {
public:

    BeatStabilityFeature() {
        quantized = true;
        track_level = false;
    }

    void compute_piece_feature(midi::Piece *x, midi::Features *f, std::string &filepath) {

        int max_beat_num = 0;
        std::map<int,double> beat_total_weights;
        std::map<std::tuple<int,int>,int> onset_weights;
        for (const auto &track : x->tracks()) {
            int total_beat_num = 0;
            for (const auto &bar : track.bars()) {
                for (const auto &event_index : bar.events()) {
                    auto event = x->events(event_index);
                    int beat_num = total_beat_num + event.time() / 12;
                    if (event.velocity()) {
                        // try extra weight for drums
                        onset_weights[std::make_tuple(beat_num,event.time() % 12)] += event.velocity(); // * (is_drum ? 2 : 1);
                        beat_total_weights[beat_num] += event.velocity();
                    }
                }
                if (abs(bar.internal_beat_length() - std::round(bar.internal_beat_length())) > 1e-4) {
                    return; // the piece is invalid and we cannot compute
                }
                total_beat_num += bar.internal_beat_length();
            }
            max_beat_num = std::max(max_beat_num, total_beat_num);
        }

        double max_weight = 0;
        for (auto &kv : beat_total_weights) {
            max_weight = std::max(max_weight, kv.second);
        }

        std::vector<double> bar_weights;
        for (int i=0; i<max_beat_num; i++) {
            auto key = std::make_tuple(i,0);
            bar_weights.push_back((onset_weights.find(key) != onset_weights.end()) ? onset_weights[key] / beat_total_weights[i] : 0);
        }

        if (bar_weights.size() == 0) {
            return;
        }

        auto fm = f->add_beat_stability();
        fm->set_filepath(filepath);
        fm->set_beat_stability_stdev(standardDeviation(bar_weights));
        fm->set_beat_stability_median(median(bar_weights));
        
    }

};


std::unique_ptr<FEATURE_EXTRACTOR> getFeature(std::string feature_name) {
    if (feature_name == "BEAT_STABILITY") return std::make_unique<BeatStabilityFeature>();
    if (feature_name == "PITCH_RANGE") return std::make_unique<PitchRangeFeature>();
    if (feature_name == "METRIC_DEPTH") return std::make_unique<MetricDepthFeature>();
    if (feature_name == "MEDIAN_METRIC_DEPTH") return std::make_unique<MedianMetricDepthFeature>();
    if (feature_name == "MOST_FREQUENT_METRIC_DEPTH") return std::make_unique<MostFrequentMetricDepthFeature>();
    if (feature_name == "DOWNBEAT_PROPORTION") return std::make_unique<DownbeatProportionFeature>();
    if (feature_name == "ALIGNED_METRIC_DEPTH") return std::make_unique<AlignedMetricDepthFeature>();
    if (feature_name == "SIMULTANEOUS_ONSET") return std::make_unique<SimultaneousOnsetFeature>();
    if (feature_name == "DRUM_PRESENCE") return std::make_unique<DrumPresenceFeature>();

    throw std::runtime_error("feature_extraction::getFeature() switch statement missing case.");
}


std::string compute_features(std::string &filepath, std::vector<std::string> &feature_names) {


    std::map<int,std::vector<std::string>> features;

    for (const auto &feature_name : feature_names) {
        features[(int)getFeature(feature_name)->quantized].push_back(feature_name);
    }

    auto encoder_config = std::make_shared<data_structures::EncoderConfig>();
    encoder_config->resolution = 12;

    midi::Features f;

    for (int i=0; i<2; i++) {
        if (features.find(i) != features.end()) {
            midi::Piece p;
            encoder_config->unquantized = 1-i;
            midi_io::ParseSong(filepath, &p, encoder_config);
            for (const auto &feature_name : features[i]) {
                getFeature(feature_name)->compute_feature(&p, &f, filepath);
            }
        }
    }

    return util_protobuf::protobuf_to_string(&f);
}


}