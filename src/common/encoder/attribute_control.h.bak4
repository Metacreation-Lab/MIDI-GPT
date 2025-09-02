#pragma once

#include <bit>
#include <map>
#include <set>

#include "representation.h"

#include "../../common/data_structures/token_sequence.h"

namespace encoder {

enum ATTRIBUTE_CONTROL_LEVEL {
    ATTRIBUTE_CONTROL_LEVEL_PIECE,
    ATTRIBUTE_CONTROL_LEVEL_TRACK,
    ATTRIBUTE_CONTROL_LEVEL_TRACK_PRE_INSTRUMENT,
    ATTRIBUTE_CONTROL_LEVEL_BAR
};

enum ATTRIBUTE_CONTROL_TRACK_TYPE {
    ATTRIBUTE_CONTROL_TRACK_TYPE_INSTRUMENT,
    ATTRIBUTE_CONTROL_TRACK_TYPE_DRUM,
    ATTRIBUTE_CONTROL_TRACK_TYPE_INSTRUMENT_AND_DRUM,
    ATTRIBUTE_CONTROL_TRACK_TYPE_NONE
};

template <typename T>
int protobuf_get_field_value(T *message, const std::string &feature_name) {
    const google::protobuf::FieldDescriptor *fd = message->GetDescriptor()->FindFieldByName(feature_name);
    if (fd == NULL) {
        throw std::runtime_error("INVALID FIELD NAME");
    }
    if (fd->type() == google::protobuf::FieldDescriptor::Type::TYPE_INT32) {
        return message->GetReflection()->GetInt32(*message, fd);
    }
    if (fd->type() == google::protobuf::FieldDescriptor::Type::TYPE_ENUM) {
        return message->GetReflection()->GetEnumValue(*message, fd);
    }
    std::cout << "field name: " << feature_name << std::endl;
    throw std::runtime_error("INVALID FIELD TYPE");
}

template <typename U, typename T>
U protobuf_get_field(const T *message, const std::string &feature_name) {
    const google::protobuf::FieldDescriptor *fd = message->GetDescriptor()->FindFieldByName(feature_name);
    if (fd == NULL) {
        throw std::runtime_error("INVALID FIELD NAME");
    }
    if (fd->type() == google::protobuf::FieldDescriptor::Type::TYPE_INT32) {
        return message->GetReflection()->GetInt32(*message, fd);
    }
    else if (fd->type() == google::protobuf::FieldDescriptor::Type::TYPE_BOOL) {
        return message->GetReflection()->GetBool(*message, fd);
    }
    else if (fd->type() == google::protobuf::FieldDescriptor::Type::TYPE_FLOAT) {
        return message->GetReflection()->GetFloat(*message, fd);
    }
    else if (fd->type() == google::protobuf::FieldDescriptor::Type::TYPE_ENUM) {
        return message->GetReflection()->GetEnumValue(*message, fd);
    }
    else {
        std::cout << "field name: " << feature_name << std::endl;
        throw std::runtime_error("INVALID FIELD TYPE");
    }
}

template <typename T, typename U>
void protobuf_set_field(T *message, const std::string &feature_name, U value) {
    const google::protobuf::FieldDescriptor *fd = message->GetDescriptor()->FindFieldByName(feature_name);
    if (fd == NULL) {
        throw std::runtime_error("INVALID FIELD NAME");
    }
    if (fd->type() == google::protobuf::FieldDescriptor::Type::TYPE_INT32) {
        message->GetReflection()->SetInt32(message, fd, value);
    }
    else if (fd->type() == google::protobuf::FieldDescriptor::Type::TYPE_BOOL) {
        message->GetReflection()->SetBool(message, fd, value);
    }
    else if (fd->type() == google::protobuf::FieldDescriptor::Type::TYPE_FLOAT) {
        message->GetReflection()->SetFloat(message, fd, value);
    }
    else if (fd->type() == google::protobuf::FieldDescriptor::Type::TYPE_ENUM) {
        message->GetReflection()->SetEnumValue(message, fd, value);
    }
    else {
        std::cout << "field name: " << feature_name << std::endl;
        throw std::runtime_error("INVALID FIELD TYPE");
    }
}

class TOKEN_COUNTER {
public:
    TOKEN_COUNTER(midi::TOKEN_TYPE tt) {
        token_type = tt;
        token_count = 0;
    }
    ~TOKEN_COUNTER() {}
    std::tuple<int,int> update(std::shared_ptr<encoder::REPRESENTATION> rep, int token) {
        bool has_changed = (rep->get_token_type(token) == token_type);
        if (has_changed) {
            token_count++;
        }
        return std::make_tuple(token_count, has_changed);
    }
    void override(int count) {
        token_count = count;
    }
    midi::TOKEN_TYPE token_type;
    int token_count;
};

class TOKEN_LABELER {
public:
    TOKEN_LABELER() {
        bar_counter = std::make_unique<TOKEN_COUNTER>(midi::TOKEN_BAR);
        track_counter = std::make_unique<TOKEN_COUNTER>(midi::TOKEN_TRACK);
    }
    ~TOKEN_LABELER() {}
    std::tuple<int,int> update(std::shared_ptr<encoder::REPRESENTATION> rep, int token) {
        auto [track_count, track_count_changed] = track_counter->update(rep,token);
        if (track_count_changed) {
            bar_counter->override(0);
        }
        auto [bar_count, bar_count_changed] = bar_counter->update(rep,token);
        return std::make_tuple(std::max(track_count-1,0),std::max(bar_count-1,0));
    }
    std::unique_ptr<TOKEN_COUNTER> bar_counter;
    std::unique_ptr<TOKEN_COUNTER> track_counter;
};

// basic implementation
std::vector<std::vector<double>> PitchProbabilityEmbedding(midi::Piece *x, std::shared_ptr<encoder::REPRESENTATION> rep, std::vector<int> &tokens) {

    // first calculate per track pitch probabilities
    std::vector<std::vector<double>> probs;
    for (const auto &track : x->tracks()) {
        double total = 0;
        std::vector<double> prob(128, 0.0);
        for (const auto &bar : track.bars()) {
            for (const auto &event_index : bar.events()) {
                if (x->events(event_index).velocity() > 0) {
                    prob[x->events(event_index).pitch()]++;
                    total++;
                }
            }
        }
        if (total > 0) {
            for (int i=0; i<128; i++) {
                prob[i] /= total;
            }
        }
        probs.push_back(prob);
    }
    
    std::vector<std::vector<double>> embeds;
    auto tl = TOKEN_LABELER();
    for (const auto &token : tokens) {
        auto [track_index, bar_index] = tl.update(rep, token);
        if (track_index >= (int)probs.size()) {
            throw std::runtime_error("INVALID TRACK INDEX DURING PitchProbabilityEmbedding()");
        }
        if (track_index < 0) {
            throw std::runtime_error("INVALID TRACK INDEX < 0 DURING PitchProbabilityEmbedding()");
        }
        embeds.push_back(probs[track_index]);
    }
    return embeds;
}

double map(double x, double in_min, double in_max, double out_min, double out_max) {
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}

class ATTRIBUTE_CONTROL {
public:

    ATTRIBUTE_CONTROL_LEVEL control_level;
    ATTRIBUTE_CONTROL_TRACK_TYPE track_type;
    std::vector<std::tuple<midi::TOKEN_TYPE,int>> token_types;
    std::vector<std::tuple<midi::TOKEN_TYPE,int,std::string>> token_types_v2;
    std::vector<std::tuple<midi::TOKEN_TYPE,int,int>> token_types_v3;
    bool precompute_on_piece;

    virtual ~ATTRIBUTE_CONTROL () {}

    virtual void compute_piece_features(midi::Piece *x, midi::PieceFeatures *pf) {
        // this function is responsible for computing the features that are needed for
        // this form of attribute control
        throw std::runtime_error("ATTRIBUTE CONTROL CLASS MUST DEFINE compute_piece_features()");
    }

    virtual void compute_track_features(midi::Piece *x, int track_num, midi::TrackFeatures *tf) {
        // this function is responsible for computing the features that are needed for
        // this form of attribute control
        throw std::runtime_error("ATTRIBUTE CONTROL CLASS MUST DEFINE compute_track_features()");
    }

    virtual void compute_bar_features(midi::Piece *x, int track_num, int bar_num, midi::BarFeatures *bf) {
        // this function is responsible for computing the features that are needed for
        // this form of attribute control
        throw std::runtime_error("ATTRIBUTE CONTROL CLASS MUST DEFINE compute_bar_features()");
    }

    virtual void append_piece_tokens(data_structures::TokenSequence *tokens, const std::shared_ptr<REPRESENTATION> &rep, midi::PieceFeatures *pf) {
        if (token_types_v2.size() > 0) {
            for (const auto &fn : token_types_v2) {
                tokens->push_back( rep->encode(std::get<0>(fn), protobuf_get_field_value(pf, std::get<2>(fn))) );
            }
        }
        else {
            throw std::runtime_error("ATTRIBUTE CONTROL MUST DEFINE append_piece_tokens()");
        }
    }

    virtual void append_track_tokens(data_structures::TokenSequence *tokens, const std::shared_ptr<REPRESENTATION> &rep, midi::TrackFeatures *tf) {
        if (token_types_v2.size() > 0) {
            for (const auto &fn : token_types_v2) {
                tokens->push_back( rep->encode(std::get<0>(fn), protobuf_get_field_value(tf, std::get<2>(fn))) );
            }
        }
        else {
            throw std::runtime_error("ATTRIBUTE CONTROL MUST DEFINE append_track_tokens()");
        }
    }

    virtual void append_bar_tokens(data_structures::TokenSequence *tokens, const std::shared_ptr<REPRESENTATION> &rep, midi::BarFeatures *bf) {
        if (token_types_v2.size() > 0) {
            for (const auto &fn : token_types_v2) {
                tokens->push_back( rep->encode(std::get<0>(fn), protobuf_get_field_value(bf, std::get<2>(fn))) );
            }
        }
        else {
            throw std::runtime_error("ATTRIBUTE CONTROL MUST DEFINE append_bar_tokens()");
        }
    }

    virtual void set_piece_mask(data_structures::TokenSequence *tokens, const std::shared_ptr<REPRESENTATION> &rep, midi::Status *piece) {
        // this function sets the appropriate token mask for sampling to control which attribute is selected
        throw std::runtime_error("ATTRIBUTE CONTROL CLASS MUST DEFINE set_piece_mask");
    }

    virtual void set_track_mask(const std::shared_ptr<REPRESENTATION> &rep, std::vector<int> &mask, midi::StatusTrack *track) {
        if (token_types_v2.size() > 0) {
            for (const auto &fn : token_types_v2) {
                rep->set_mask(std::get<0>(fn), {protobuf_get_field_value(track, std::get<2>(fn))-1}, mask, 1);
            }
        }
        else {
            throw std::runtime_error("ATTRIBUTE CONTROL CLASS MUST DEFINE set_track_mask");
        }
    }

    virtual void set_bar_mask(const std::shared_ptr<REPRESENTATION> &rep, std::vector<int> &mask, midi::StatusBar *bar) {
        if (token_types_v2.size() > 0) {
            for (const auto &fn : token_types_v2) {
                rep->set_mask(std::get<0>(fn), {protobuf_get_field_value(bar, std::get<2>(fn))-1}, mask, 1);
            }
        }
        else {
            throw std::runtime_error("ATTRIBUTE CONTROL CLASS MUST DEFINE set_bar_mask");
        }
    }

    virtual void override_track_feature(midi::TrackFeatures *tf, midi::StatusTrack *track) {
        if (token_types_v2.size() > 0) {
            for (const auto &fn : token_types_v2) {
                auto value = protobuf_get_field_value(track, std::get<2>(fn));
                if (value > 0) {
                    protobuf_set_field(tf, std::get<2>(fn), value - 1); // copy value from status to piece
                }
            }
        }
        else {
            throw std::runtime_error("ATTRIBUTE CONTROL CLASS MUST DEFINE override_track_feature");
        }
    }

    virtual void override_bar_feature(midi::BarFeatures *bf, midi::StatusBar *bar) {
        if (token_types_v2.size() > 0) {
            for (const auto &fn : token_types_v2) {
                auto value = protobuf_get_field_value(bar, std::get<2>(fn));
                if (value > 0) {
                    protobuf_set_field(bf, std::get<2>(fn), value - 1); // copy value from status to piece
                }
            }
        }
        else {
            throw std::runtime_error("ATTRIBUTE CONTROL CLASS MUST DEFINE override_bar_feature");
        }
    }

    void override_track_level_features(midi::Piece *x, midi::Status *s) {
        for (int track_num=0; track_num<x->tracks_size(); track_num++) {
            midi::TrackFeatures *tf = util_protobuf::GetTrackFeatures(x,track_num);
            midi::StatusTrack st = s->tracks(track_num);
            override_track_feature(tf, &st);
        }
    }

    void override_bar_level_features(midi::Piece *x, midi::Status *s) {
        for (int track_num=0; track_num<x->tracks_size(); track_num++) {
            midi::Track *track = x->mutable_tracks(track_num);
            midi::StatusTrack st = s->tracks(track_num);
            for (int bar_num=0; bar_num<track->bars_size(); bar_num++) {
                midi::BarFeatures *bf = util_protobuf::GetBarFeatures(track, bar_num);
                midi::StatusBar sb = st.bars(bar_num);
                override_bar_feature(bf, &sb);
            }
        }
    }

    void override_features(midi::Piece *x, midi::Status *s) {
        switch(control_level) {
            case ATTRIBUTE_CONTROL_LEVEL_PIECE:
                throw std::runtime_error("CANNOT OVERRIDE PIECE LEVEL FEATURES");
                break;
            case ATTRIBUTE_CONTROL_LEVEL_TRACK:
                override_track_level_features(x,s);
                break;
            case ATTRIBUTE_CONTROL_LEVEL_TRACK_PRE_INSTRUMENT:
                override_track_level_features(x,s);
                break;
            case ATTRIBUTE_CONTROL_LEVEL_BAR:
                override_bar_level_features(x,s);
                break;
            default:
                throw std::runtime_error("INVALID ATTRIBUTE CONTROL LEVEL");
        }
    }

    void compute_piece_level_features(midi::Piece *x) {
        midi::PieceFeatures *pf = util_protobuf::GetPieceFeatures(x);
        compute_piece_features(x, pf);
    }

    void compute_track_level_features(midi::Piece *x) {
        for (int track_num=0; track_num<x->tracks_size(); track_num++) {
            midi::TrackFeatures *tf = util_protobuf::GetTrackFeatures(x,track_num);
            compute_track_features(x, track_num, tf);
        }
    }

    void compute_bar_level_features(midi::Piece *x) {
        for (int track_num=0; track_num<x->tracks_size(); track_num++) {
            midi::Track *track = x->mutable_tracks(track_num);
            for (int bar_num=0; bar_num<track->bars_size(); bar_num++) {
                midi::BarFeatures *bf = util_protobuf::GetBarFeatures(track, bar_num);
                compute_bar_features(x, track_num, bar_num, bf);
            }
        }
    }

    void compute_features(midi::Piece *x) {
        switch(control_level) {
            case ATTRIBUTE_CONTROL_LEVEL_PIECE:
                compute_piece_level_features(x);
                break;
            case ATTRIBUTE_CONTROL_LEVEL_TRACK:
                compute_track_level_features(x);
                break;
            case ATTRIBUTE_CONTROL_LEVEL_TRACK_PRE_INSTRUMENT:
                compute_track_level_features(x);
                break;
            case ATTRIBUTE_CONTROL_LEVEL_BAR:
                compute_bar_level_features(x);
                break;
            default:
                throw std::runtime_error("INVALID ATTRIBUTE CONTROL LEVEL");
        }
    }

    virtual double evaluate_track_feature(midi::Piece *x, int track_num, midi::TrackFeatures *tf, midi::StatusTrack *st) {
        throw std::runtime_error("ATTRIBUTE CONTROL CLASS MUST DEFINE evaluate_track_feature()");
    }

    std::vector<double> evaluate_track_feature_py(std::string &piece_json, std::string &status_json) {
        midi::Piece x;
        midi::Status s;
        util_protobuf::string_to_protobuf(piece_json, &x);
        util_protobuf::string_to_protobuf(status_json, &s);
        std::vector<double> output;
        for (int i=0; i<x.tracks_size(); i++) {
            output.push_back( evaluate_track_feature(&x, i, util_protobuf::GetTrackFeatures(&x,i), s.mutable_tracks(i)) );
        }
        return output;
    }

    std::string compute_features_py(std::string &piece_json) {
        midi::Piece x;
        util_protobuf::string_to_protobuf(piece_json, &x);
        compute_features(&x);
        return util_protobuf::protobuf_to_string(&x);
    }

    std::string compute_piece_level_features_py(std::string &piece_json) {
        midi::Piece x;
        util_protobuf::string_to_protobuf(piece_json, &x);
        compute_piece_level_features(&x);
        return util_protobuf::protobuf_to_string(&x);
    }

    bool check_valid_track(bool is_drum) {
        if ((track_type == ATTRIBUTE_CONTROL_TRACK_TYPE_DRUM) && (is_drum)) {
            return true;
        }
        if ((track_type == ATTRIBUTE_CONTROL_TRACK_TYPE_INSTRUMENT) && (!is_drum)) {
            return true;
        }
        if (track_type == ATTRIBUTE_CONTROL_TRACK_TYPE_INSTRUMENT_AND_DRUM) {
            return true;
        }
        return false;
    }

    std::vector<midi::TOKEN_TYPE> get_token_types() {
        std::vector<midi::TOKEN_TYPE> token_types_list;
        for (const auto &ttd : token_types) {
            token_types_list.push_back(std::get<0>(ttd));
        }
        return token_types_list;
    }

    int get_token_domain_size(midi::TOKEN_TYPE tt) {
        for (const auto &ttd : token_types) {
            if (std::get<0>(ttd) == tt) {
                return std::get<1>(ttd);
            }
        }
        throw std::runtime_error("ATTRIBUTE_CONTROL::get_token_domain_size() : TOKEN TYPE NOT FOUND");
    }

    virtual TOKEN_DOMAIN get_token_domain(midi::TOKEN_TYPE tt) {
        return TOKEN_DOMAIN(get_token_domain_size(tt));
    }

    bool is_track_control() {
        return (control_level == ATTRIBUTE_CONTROL_LEVEL_TRACK) || (control_level == ATTRIBUTE_CONTROL_LEVEL_TRACK_PRE_INSTRUMENT);
    }

    bool is_bar_control() {
        return (control_level == ATTRIBUTE_CONTROL_LEVEL_BAR);
    }


    // get the enum domain for the attribute control in status track
    std::map<std::string,std::vector<std::string>> get_status_track_enum_domain() {
        if (token_types_v2.size() == 0) {
            throw std::runtime_error("STATUS TRACK FIELD NAME NOT SPECIFIED");
        }
        midi::StatusTrack st;
        midi::StatusBar sb;
        const google::protobuf::Descriptor *descriptor = is_bar_control() ? sb.GetDescriptor() : st.GetDescriptor();
        if (descriptor == NULL) {
            throw std::runtime_error("INVALID DESCRIPTOR");
        }
        std::map<std::string,std::vector<std::string>> output;
        for (const auto &fn : token_types_v2) {
            //std::cout << "FIELD NAME: " << std::get<2>(fn) << std::endl;
            auto field_name = std::get<2>(fn);
            const google::protobuf::FieldDescriptor *field = descriptor->FindFieldByName(field_name);
            if (field == NULL) {
                throw std::runtime_error("INVALID FIELD NAME");
            }
            auto enum_descriptor = field->enum_type();
            if (enum_descriptor == NULL) {
                throw std::runtime_error("INVALID ENUM TYPE");
            }
            for (int i=0; i<enum_descriptor->value_count(); i++) {
                output[field_name].push_back(enum_descriptor->value(i)->name());
            }   
        }        
		return output;
    }

    std::map<std::string,std::map<std::string,int>> get_status_enum_mapping() {
        if (token_types_v2.size() == 0) {
            throw std::runtime_error("STATUS BAR FIELD NAME NOT SPECIFIED");
        }
        midi::StatusTrack st;
        midi::StatusBar sb;
        const google::protobuf::Descriptor *descriptor = is_bar_control() ? sb.GetDescriptor() : st.GetDescriptor();
        if (descriptor == NULL) {
            throw std::runtime_error("INVALID DESCRIPTOR");
        }
        std::map<std::string,std::map<std::string,int>> output;
        for (const auto &fn : token_types_v2) {
            auto field_name = std::get<2>(fn);
            const google::protobuf::FieldDescriptor *field = descriptor->FindFieldByName(field_name);
            if (field == NULL) {
                throw std::runtime_error("INVALID FIELD NAME");
            }
            auto enum_descriptor = field->enum_type();
            for (int i=0; i<enum_descriptor->value_count(); i++) {
                output[field_name][enum_descriptor->value(i)->name()] = i;
            }   
        }
        return output;
    }

};

// ================================================
// ================================================
// ATTRIBUTE CONTROLS
// ================================================
// ================================================

class TrackLevelOnsetPolyphony : public ATTRIBUTE_CONTROL {
public:

    TrackLevelOnsetPolyphony() {
        precompute_on_piece = false;
        control_level = ATTRIBUTE_CONTROL_LEVEL_TRACK;
        track_type = ATTRIBUTE_CONTROL_TRACK_TYPE_INSTRUMENT_AND_DRUM;
        token_types = {
            {midi::TOKEN_TRACK_LEVEL_ONSET_POLYPHONY_MIN, 6},
            {midi::TOKEN_TRACK_LEVEL_ONSET_POLYPHONY_MAX, 6}
        };
        token_types_v2 = {
            {midi::TOKEN_TRACK_LEVEL_ONSET_POLYPHONY_MIN, 6, "onset_polyphony_min"},
            {midi::TOKEN_TRACK_LEVEL_ONSET_POLYPHONY_MAX, 6, "onset_polyphony_max"}
        };
    }
    ~TrackLevelOnsetPolyphony() {}

    void compute_track_features(midi::Piece *x, int track_num, midi::TrackFeatures *tf) {
        const auto track = x->tracks(track_num);
        tf->mutable_attribute_control_distributions()->clear_onset_polyphony();

        int bar_start = 0;
        std::map<int,int> concurrent_onsets;
        for (const auto &bar : track.bars()) {
            for (const auto &event_index : bar.events()) {
                if (x->events(event_index).velocity()) {
                    concurrent_onsets[bar_start + x->events(event_index).time()] += 1;
                }
            }
            bar_start += x->resolution() * bar.internal_beat_length();
        }

        int polyphony_min = INT_MAX;
        int polyphony_max = INT_MIN;
        for (const auto &kv : concurrent_onsets) {
            if (kv.second < polyphony_min) {
                polyphony_min = kv.second;
            }
            if (kv.second > polyphony_max) {
                polyphony_max = kv.second;
            }
            tf->mutable_attribute_control_distributions()->add_onset_polyphony(kv.second); // for evaluation
        }
        
        tf->set_onset_polyphony_min( util_protobuf::clip(polyphony_min, 1, get_token_domain_size(midi::TOKEN_TRACK_LEVEL_ONSET_POLYPHONY_MIN)) - 1 );
        tf->set_onset_polyphony_max( util_protobuf::clip(polyphony_max, 1, get_token_domain_size(midi::TOKEN_TRACK_LEVEL_ONSET_POLYPHONY_MAX)) - 1 );
    }


    double evaluate_track_feature(midi::Piece *x, int track_num, midi::TrackFeatures *tf, midi::StatusTrack *st) {
        compute_track_features(x, track_num, tf);
        auto mapping = get_status_enum_mapping();
        auto domain = get_status_track_enum_domain();
        double range_min = mapping["onset_polyphony_min"][domain["onset_polyphony_min"][protobuf_get_field_value(st, "onset_polyphony_min")]];
        double range_max = mapping["onset_polyphony_max"][domain["onset_polyphony_max"][protobuf_get_field_value(st, "onset_polyphony_max")]];
        double score = 0.0;
        double total = 0.0;
        for (const auto value : tf->attribute_control_distributions().onset_polyphony()) {
            score += (range_min <= value) && (value <= range_max);
            total += 1;
        }
        return score / total;
    }
};


class TrackLevelNoteDuration : public ATTRIBUTE_CONTROL {
public:

    TrackLevelNoteDuration() {
        precompute_on_piece = false;
        control_level = ATTRIBUTE_CONTROL_LEVEL_TRACK;
        track_type = ATTRIBUTE_CONTROL_TRACK_TYPE_INSTRUMENT;
        token_types = {
            {midi::TOKEN_CONTAINS_NOTE_DURATION_THIRTY_SECOND, 2},
            {midi::TOKEN_CONTAINS_NOTE_DURATION_SIXTEENTH, 2},
            {midi::TOKEN_CONTAINS_NOTE_DURATION_EIGHTH, 2},
            {midi::TOKEN_CONTAINS_NOTE_DURATION_QUARTER, 2},
            {midi::TOKEN_CONTAINS_NOTE_DURATION_HALF, 2},
            {midi::TOKEN_CONTAINS_NOTE_DURATION_WHOLE, 2}
        };
        token_types_v2 = {
            {midi::TOKEN_CONTAINS_NOTE_DURATION_THIRTY_SECOND, 2, "contains_note_duration_thirty_second"},
            {midi::TOKEN_CONTAINS_NOTE_DURATION_SIXTEENTH, 2, "contains_note_duration_sixteenth"},
            {midi::TOKEN_CONTAINS_NOTE_DURATION_EIGHTH, 2, "contains_note_duration_eighth"},
            {midi::TOKEN_CONTAINS_NOTE_DURATION_QUARTER, 2, "contains_note_duration_quarter"},
            {midi::TOKEN_CONTAINS_NOTE_DURATION_HALF, 2, "contains_note_duration_half"},
            {midi::TOKEN_CONTAINS_NOTE_DURATION_WHOLE, 2, "contains_note_duration_whole"}
        };
    }
    ~TrackLevelNoteDuration() {}

    void compute_track_features(midi::Piece *x, int track_num, midi::TrackFeatures *tf) {
        // add in the note duration distribution for testing at some point ...
        const auto track = x->tracks(track_num);
        tf->mutable_attribute_control_distributions()->note_duration();

        int max_tick = 0;
        std::vector<midi::Note> notes = util_protobuf::TrackEventsToNotes(x, track_num, &max_tick);

        // get note durations
        std::vector<int> durations;
		for (const auto &note : notes) {
			double d = note.end() - note.start();
            int duration_level = (int)util_protobuf::clip(util_protobuf::midigpt_log2(std::max(d / 3., 1e-6)), 0., 5.); // assume resolution==24
			durations.push_back(duration_level);
            tf->mutable_attribute_control_distributions()->add_note_duration(duration_level); // for evaluation
		}

        // see which categories are used
        std::vector<int> used_categories(6, 0);
        for (const auto &d : durations) {
            used_categories[d] = 1;
        }
		
        // add features
        tf->set_contains_note_duration_thirty_second(used_categories[0]);
        tf->set_contains_note_duration_sixteenth(used_categories[1]);
        tf->set_contains_note_duration_eighth(used_categories[2]);
        tf->set_contains_note_duration_quarter(used_categories[3]);
        tf->set_contains_note_duration_half(used_categories[4]);
        tf->set_contains_note_duration_whole(used_categories[5]);
    }

    double evaluate_track_feature(midi::Piece *x, int track_num, midi::TrackFeatures *tf, midi::StatusTrack *st) {
        compute_track_features(x, track_num, tf);
        std::map<int,std::string> mapping = {
            {0,"contains_note_duration_thirty_second"},
            {1,"contains_note_duration_sixteenth"},
            {2,"contains_note_duration_eighth"},
            {3,"contains_note_duration_quarter"},
            {4,"contains_note_duration_half"},
            {5,"contains_note_duration_whole"}
        };
        double score = 0.0;
        double total = 0.0;
        const google::protobuf::Reflection *reflection = st->GetReflection();
        const google::protobuf::Descriptor *descriptor = st->GetDescriptor();
        for (const auto value : tf->attribute_control_distributions().note_duration()) {
            const google::protobuf::FieldDescriptor *fd = descriptor->FindFieldByName(mapping[value]);
            score += (reflection->GetEnumValue(*st, fd) == midi::BOOLEAN_TRUE);
            total += 1;
        }
        return score / total;
    }
};

class TrackLevelOnsetDensity : public ATTRIBUTE_CONTROL {
public:

    TrackLevelOnsetDensity() {
        precompute_on_piece = false;
        control_level = ATTRIBUTE_CONTROL_LEVEL_TRACK;
        track_type = ATTRIBUTE_CONTROL_TRACK_TYPE_INSTRUMENT_AND_DRUM;
        token_types = {
            {midi::TOKEN_TRACK_LEVEL_ONSET_DENSITY_MIN, 18},
            {midi::TOKEN_TRACK_LEVEL_ONSET_DENSITY_MAX, 18}
        };
        token_types_v2 = {
            {midi::TOKEN_TRACK_LEVEL_ONSET_DENSITY_MIN, 18, "onset_density_min"},
            {midi::TOKEN_TRACK_LEVEL_ONSET_DENSITY_MAX, 18, "onset_density_max"}
        };
    }
    ~TrackLevelOnsetDensity() {}

    void compute_track_features(midi::Piece *x, int track_num, midi::TrackFeatures *tf) {
        const auto track = x->tracks(track_num);
        tf->mutable_attribute_control_distributions()->clear_onset_density();

        std::vector<int> unique_onsets_per_bar;
        for (const auto &bar : track.bars()) {
            std::set<int> unique_onsets;
            for (const auto &event_index : bar.events()) {
                if (x->events(event_index).velocity()) {
                    unique_onsets.insert(x->events(event_index).time());
                }
            }
            unique_onsets_per_bar.push_back( util_protobuf::clip((int)unique_onsets.size(), 0, get_token_domain_size(midi::TOKEN_TRACK_LEVEL_ONSET_DENSITY_MIN)-1) ); // 18 classes
        }

        int onsets_min = INT_MAX;
        int onsets_max = INT_MIN;
        for (const auto &x : unique_onsets_per_bar) {
            if (x < onsets_min) {
                onsets_min = x;
            }
            if (x > onsets_max) {
                onsets_max = x;
            }
            tf->mutable_attribute_control_distributions()->add_onset_density(x); // for evaluation
        }

        tf->set_onset_density_min( onsets_min );
        tf->set_onset_density_max( onsets_max );
    }

    double evaluate_track_feature(midi::Piece *x, int track_num, midi::TrackFeatures *tf, midi::StatusTrack *st) {
        compute_track_features(x, track_num, tf);
        auto mapping = get_status_enum_mapping();
        auto domain = get_status_track_enum_domain();
        double range_min = mapping["onset_density_min"][domain["onset_density_min"][protobuf_get_field_value(st, "onset_density_min")]];
        double score = 0.0;
        double total = 0.0;
        for (const auto value : tf->attribute_control_distributions().onset_density()) {
            score += abs(value - range_min);
            total += 1;
        }
        return score / total;
    }
};

class BarLevelOnsetPolyphony : public ATTRIBUTE_CONTROL {
public:

    BarLevelOnsetPolyphony() {
        precompute_on_piece = false;
        control_level = ATTRIBUTE_CONTROL_LEVEL_BAR;
        track_type = ATTRIBUTE_CONTROL_TRACK_TYPE_INSTRUMENT_AND_DRUM;
        token_types = {
            {midi::TOKEN_BAR_LEVEL_ONSET_POLYPHONY_MIN, 6},
            {midi::TOKEN_BAR_LEVEL_ONSET_POLYPHONY_MAX, 6}
        };
        token_types_v2 = {
            {midi::TOKEN_BAR_LEVEL_ONSET_POLYPHONY_MIN, 6, "onset_polyphony_min"},
            {midi::TOKEN_BAR_LEVEL_ONSET_POLYPHONY_MAX, 6, "onset_polyphony_max"}
        };
    }
    ~BarLevelOnsetPolyphony() {}

    void compute_bar_features(midi::Piece *x, int track_num, int bar_num, midi::BarFeatures *bf) {
        const auto track = x->tracks(track_num);
        const auto bar = track.bars(bar_num);

        std::map<int,int> concurrent_onsets;
        for (const auto &event_index : bar.events()) {
            if (x->events(event_index).velocity()) {
                concurrent_onsets[x->events(event_index).time()] += 1;
            }
        }

        // get the min and max of concurrent onsets
        int polyphony_min = INT_MAX;
        int polyphony_max = INT_MIN;
        for (const auto &kv : concurrent_onsets) {
            if (kv.second < polyphony_min) {
                polyphony_min = kv.second;
            }
            if (kv.second > polyphony_max) {
                polyphony_max = kv.second;
            }
        }
        
        bf->set_onset_polyphony_min( util_protobuf::clip(
            polyphony_min, 1, get_token_domain_size(midi::TOKEN_BAR_LEVEL_ONSET_POLYPHONY_MIN)) - 1 );
        bf->set_onset_polyphony_max( util_protobuf::clip(
            polyphony_max, 1, get_token_domain_size(midi::TOKEN_BAR_LEVEL_ONSET_POLYPHONY_MAX)) - 1 );
    }
};

class BarLevelOnsetDensity : public ATTRIBUTE_CONTROL {
public:

    BarLevelOnsetDensity() {
        precompute_on_piece = false;
        control_level = ATTRIBUTE_CONTROL_LEVEL_BAR;
        track_type = ATTRIBUTE_CONTROL_TRACK_TYPE_INSTRUMENT_AND_DRUM;
        token_types = {
            {midi::TOKEN_BAR_LEVEL_ONSET_DENSITY, 18}
        };
        token_types_v2 = {
            {midi::TOKEN_BAR_LEVEL_ONSET_DENSITY, 18, "onset_density"}
        };
    }
    ~BarLevelOnsetDensity() {}

    void compute_bar_features(midi::Piece *x, int track_num, int bar_num, midi::BarFeatures *bf) {
        const auto track = x->tracks(track_num);
        const auto bar = track.bars(bar_num);

        std::set<int> unique_onsets;
        for (const auto &event_index : bar.events()) {
            if (x->events(event_index).velocity()) {
                unique_onsets.insert(x->events(event_index).time());
            }
        }
        
        bf->set_onset_density(util_protobuf::clip(
            (int)unique_onsets.size(), 0, get_token_domain_size(midi::TOKEN_BAR_LEVEL_ONSET_DENSITY)-1));
    }
};

class PolyphonyQuantile : public ATTRIBUTE_CONTROL {
public:

    PolyphonyQuantile() {
        precompute_on_piece = false;
        control_level = ATTRIBUTE_CONTROL_LEVEL_TRACK;
        track_type = ATTRIBUTE_CONTROL_TRACK_TYPE_INSTRUMENT;
        token_types = {
            {midi::TOKEN_MIN_POLYPHONY, 10},
            {midi::TOKEN_MAX_POLYPHONY, 10}
        };
        token_types_v2 = {
            {midi::TOKEN_MIN_POLYPHONY, 10, "min_polyphony_q"},
            {midi::TOKEN_MAX_POLYPHONY, 10, "max_polyphony_q"}
        };
    }
    ~PolyphonyQuantile() {}

    void compute_track_features(midi::Piece *x, int track_num, midi::TrackFeatures *tf) {
        const auto track = x->tracks(track_num);
        tf->mutable_attribute_control_distributions()->clear_polyphony_quantile();

        int max_tick = 0;
        std::vector<midi::Note> notes = util_protobuf::TrackEventsToNotes(x, track_num, &max_tick);
		int nonzero_count = 0;
		double count = 0;
		std::vector<int> flat_roll(max_tick, 0);
		for (const auto &note : notes) {
			for (int t = note.start(); t < std::min(note.end(), max_tick - 1); t++) {
				if (flat_roll[t] == 0) {
					nonzero_count += 1;
				}
				flat_roll[t]++;
				count++;
			}
		}

		std::vector<int> nz;
		for (const auto &x : flat_roll) {
			if (x > 0) {
				nz.push_back(x);
                tf->mutable_attribute_control_distributions()->add_polyphony_quantile(x); // for evaluation
			}
		}

        // get quantiles and add to track features
		std::vector<int> polyphony_qs = util_protobuf::quantile<int>(nz, { .15,.85 });
        tf->set_min_polyphony_q( util_protobuf::clip(polyphony_qs[0], 1, get_token_domain_size(midi::TOKEN_MIN_POLYPHONY)) - 1 );
        tf->set_max_polyphony_q( util_protobuf::clip(polyphony_qs[1], 1, get_token_domain_size(midi::TOKEN_MAX_POLYPHONY)) - 1 );
    }
};

class NoteDurationQuantile : public ATTRIBUTE_CONTROL {
public:

    NoteDurationQuantile() {
        precompute_on_piece = false;
        control_level = ATTRIBUTE_CONTROL_LEVEL_TRACK;
        track_type = ATTRIBUTE_CONTROL_TRACK_TYPE_INSTRUMENT;
        token_types = {
            {midi::TOKEN_MIN_NOTE_DURATION, 6},
            {midi::TOKEN_MAX_NOTE_DURATION, 6}
        };
        token_types_v2 = {
            {midi::TOKEN_MIN_NOTE_DURATION, 6, "min_note_duration_q"},
            {midi::TOKEN_MAX_NOTE_DURATION, 6, "max_note_duration_q"}
        };
    }
    ~NoteDurationQuantile() {}

    void compute_track_features(midi::Piece *x, int track_num, midi::TrackFeatures *tf) {
        const auto track = x->tracks(track_num);
        tf->mutable_attribute_control_distributions()->clear_note_duration_quantile();

        int max_tick = 0;
        std::vector<midi::Note> notes = util_protobuf::TrackEventsToNotes(x, track_num, &max_tick);

        // get note durations
        std::vector<int> durations;
		for (const auto &note : notes) {
			double d = note.end() - note.start();
            int duration_level = (int)util_protobuf::clip(util_protobuf::midigpt_log2(std::max(d / 3., 1e-6)) + 1, 0., (double)get_token_domain_size(midi::TOKEN_MIN_NOTE_DURATION)-1.);
			durations.push_back(duration_level);
            tf->mutable_attribute_control_distributions()->add_note_duration_quantile(duration_level); // for evaluation
		}
		
        // get quantiles and add to track features
        std::vector<int> dur_qs = util_protobuf::quantile(durations, { .15,.85 });
		tf->set_min_note_duration_q(dur_qs[0]);
		tf->set_max_note_duration_q(dur_qs[1]);
    }
};

class NoteDensity : public ATTRIBUTE_CONTROL {
public:

    NoteDensity() {
        precompute_on_piece = false;
        control_level = ATTRIBUTE_CONTROL_LEVEL_TRACK;
        track_type = ATTRIBUTE_CONTROL_TRACK_TYPE_DRUM;
        token_types = {
            {midi::TOKEN_DENSITY_LEVEL, 10}
        };
        token_types_v2 = {
            {midi::TOKEN_DENSITY_LEVEL, 10, "note_density_level"}
        };
    }
    ~NoteDensity() {}

    void compute_track_features(midi::Piece *x, int track_num, midi::TrackFeatures *tf) {
        const auto track = x->tracks(track_num);

        // calculate average notes per bar
        int num_notes = 0;
        int bar_num = 0;
        std::set<int> valid_bars;
        for (const auto &bar : track.bars()) {
            for (const auto &event_index : bar.events()) {
                if (x->events(event_index).velocity()) {
                    valid_bars.insert(bar_num);
                    num_notes++;
                }
            }
            bar_num++;
        }
        int num_bars = std::max((int)valid_bars.size(), 1);
        double av_notes_fp = (double)num_notes / num_bars;
        int av_notes = round(av_notes_fp);

        // calculate the density bin
        int qindex = track.instrument();
        int bin = 0;

        if (data_structures::is_drum_track(track.track_type())) {
            qindex = 128;
        }
        while (av_notes > enums::DENSITY_QUANTILES[qindex][bin]) {
            bin++;
        }

        tf->set_note_density_level(bin);
        tf->set_note_density_value(av_notes_fp);
    }
};

template <typename T>
T median(std::vector<T> &xs) {
    std::sort(xs.begin(), xs.end());
    return xs[xs.size() / 2];
}

class PitchRange : public ATTRIBUTE_CONTROL {
public:

    PitchRange() {
        precompute_on_piece = false;
        control_level = ATTRIBUTE_CONTROL_LEVEL_TRACK;
        track_type = ATTRIBUTE_CONTROL_TRACK_TYPE_INSTRUMENT;
        token_types = {
            {midi::TOKEN_TRACK_LEVEL_PITCH_RANGE_MIN, 128},
            {midi::TOKEN_TRACK_LEVEL_PITCH_RANGE_MAX, 128}
        };
    }
    ~PitchRange() {}

    void compute_track_features(midi::Piece *x, int track_num, midi::TrackFeatures *tf) {
        const auto track = x->tracks(track_num);
        int min_pitch = 127;
        int max_pitch = 0;
        for (const auto &bar : track.bars()) {
            for (const auto &event_index : bar.events()) {
                if (x->events(event_index).velocity()) {
                    int pitch = x->events(event_index).pitch();
                    if (pitch < min_pitch) {
                        min_pitch = pitch;
                    }
                    if (pitch > max_pitch) {
                        max_pitch = pitch;
                    }
                }
            }
        }
        tf->set_min_pitch(min_pitch);
        tf->set_max_pitch(max_pitch);
    }

    void append_track_tokens(data_structures::TokenSequence *tokens, const std::shared_ptr<REPRESENTATION> &rep, midi::TrackFeatures *tf) {
        tokens->push_back( rep->encode(midi::TOKEN_TRACK_LEVEL_PITCH_RANGE_MIN, tf->min_pitch()) );
        tokens->push_back( rep->encode(midi::TOKEN_TRACK_LEVEL_PITCH_RANGE_MAX, tf->max_pitch()) );
    }

    void set_track_mask(const std::shared_ptr<REPRESENTATION> &rep, std::vector<int> &mask, midi::StatusTrack *track) {
        rep->set_mask(midi::TOKEN_TRACK_LEVEL_PITCH_RANGE_MIN, {track->min_pitch()}, mask, 1);
        rep->set_mask(midi::TOKEN_TRACK_LEVEL_PITCH_RANGE_MAX, {track->max_pitch()}, mask, 1);
    }
};

class Genre : public ATTRIBUTE_CONTROL {
public:

    Genre() {
        precompute_on_piece = false;
        control_level = ATTRIBUTE_CONTROL_LEVEL_TRACK_PRE_INSTRUMENT;
        track_type = ATTRIBUTE_CONTROL_TRACK_TYPE_INSTRUMENT_AND_DRUM;
        token_types = {
            {midi::TOKEN_GENRE, static_cast<int>(midi::GENRE_MUSICMAP_NONE)}
        };
        token_types_v2 = {
            {midi::TOKEN_GENRE, static_cast<int>(midi::GENRE_MUSICMAP_NONE), "genre"}
        };
    }
    ~Genre() {}

    void compute_track_features(midi::Piece *x, int track_num, midi::TrackFeatures *tf) {
        auto metadata_label = x->internal_metadata_labels().genre();
        if (metadata_label == midi::GENRE_MUSICMAP_ANY) {
            metadata_label = midi::GENRE_MUSICMAP_NONE;
        }
        tf->set_genre(static_cast<int>(metadata_label)-1);
    }

    // override get token domain to get the different strings
    TOKEN_DOMAIN get_token_domain(midi::TOKEN_TYPE tt) {
        if (tt != midi::TOKEN_GENRE) {
            throw std::runtime_error("Genre::get_token_domain: invalid token type");
        }
        std::vector<std::string> domain;
        for (int i=0; i<midi::GENRE_MUSICMAP_NONE; i++) {
            const google::protobuf::EnumDescriptor *descriptor = midi::GenreMusicmap_descriptor();
            std::string name = descriptor->FindValueByNumber(static_cast<midi::GenreMusicmap>(i+1))->name();
            domain.push_back(name);
        }
        return TOKEN_DOMAIN(domain, STRING_VALUES_DOMAIN);
    }
};


// ================================================
// ================================================
// ATTRIBUTE CONTROL HELPERS
// ================================================
// ================================================

std::unique_ptr<ATTRIBUTE_CONTROL> getAttributeControl(midi::ATTRIBUTE_CONTROL_TYPE ac_type) {
    switch(ac_type) {
        case midi::ATTRIBUTE_CONTROL_NOTE_DENSITY: return std::make_unique<NoteDensity>();
        case midi::ATTRIBUTE_CONTROL_TRACK_LEVEL_ONSET_POLYPHONY: return std::make_unique<TrackLevelOnsetPolyphony>();
        case midi::ATTRIBUTE_CONTROL_TRACK_LEVEL_ONSET_DENSITY: return std::make_unique<TrackLevelOnsetDensity>();
        case midi::ATTRIBUTE_CONTROL_PITCH_RANGE: return std::make_unique<PitchRange>();
        case midi::ATTRIBUTE_CONTROL_GENRE: return std::make_unique<Genre>();

        case midi::ATTRIBUTE_CONTROL_TRACK_LEVEL_NOTE_DURATION: return std::make_unique<TrackLevelNoteDuration>();

        case midi::ATTRIBUTE_CONTROL_POLYPHONY_QUANTILE: return std::make_unique<PolyphonyQuantile>();
        case midi::ATTRIBUTE_CONTROL_NOTE_DURATION_QUANTILE: return std::make_unique<NoteDurationQuantile>();

        case midi::ATTRIBUTE_CONTROL_BAR_LEVEL_ONSET_DENSITY: return std::make_unique<BarLevelOnsetDensity>();
        case midi::ATTRIBUTE_CONTROL_BAR_LEVEL_ONSET_POLYPHONY: return std::make_unique<BarLevelOnsetPolyphony>();
        case midi::ATTRIBUTE_CONTROL_END:
            throw std::runtime_error("encoder::getAttributeControl() midi::ATTRIBUTE_CONTROL_END is an invalid argument."); 
    }
    throw std::runtime_error("encoder::getAttributeControl() switch statement missing case.");
}

std::unique_ptr<ATTRIBUTE_CONTROL> getAttributeControlStr(std::string &ac_type) {
    auto descriptor = google::protobuf::GetEnumDescriptor<midi::ATTRIBUTE_CONTROL_TYPE>();
    auto value_descriptor = descriptor->FindValueByName(ac_type);
    if (value_descriptor == NULL) {
        throw std::runtime_error("encoder::getAttributeControlStr() invalid attribute control type.");
    }
    return getAttributeControl(static_cast<midi::ATTRIBUTE_CONTROL_TYPE>(value_descriptor->index()));
}

std::vector<std::unique_ptr<ATTRIBUTE_CONTROL>> getAttributeControls() {
    std::vector<std::unique_ptr<ATTRIBUTE_CONTROL>> acs;
    for(int i=0; i<midi::ATTRIBUTE_CONTROL_END; i++){
        acs.push_back(getAttributeControl(static_cast<midi::ATTRIBUTE_CONTROL_TYPE>(i)));
    }
    return acs;
}

std::vector<midi::TOKEN_TYPE> getAttributeControlTokenTypes() {
    std::vector<midi::TOKEN_TYPE> token_types;
    for (const auto &ac : getAttributeControls()) {
        token_types.push_back(ac->get_token_types()[0]);
    }
    return token_types;
}

std::map<midi::TOKEN_TYPE,midi::ATTRIBUTE_CONTROL_TYPE> getTokenToAttributeControlTypeMap() {
    std::map<midi::TOKEN_TYPE,midi::ATTRIBUTE_CONTROL_TYPE> token_to_ac_type;
    for(int i=0; i<midi::ATTRIBUTE_CONTROL_END; i++){
        auto ac_type = static_cast<midi::ATTRIBUTE_CONTROL_TYPE>(i);
        auto ac = getAttributeControl(ac_type);
        token_to_ac_type[ac->get_token_types()[0]] = ac_type;
    }
    return token_to_ac_type;
}

std::multimap<midi::TOKEN_TYPE,midi::ATTRIBUTE_CONTROL_TYPE> getTokenToAttributeControlTypeMultimap() {
    std::multimap<midi::TOKEN_TYPE,midi::ATTRIBUTE_CONTROL_TYPE> token_to_ac_type;
    for(int i=0; i<midi::ATTRIBUTE_CONTROL_END; i++){
        auto ac_type = static_cast<midi::ATTRIBUTE_CONTROL_TYPE>(i);
        auto ac = getAttributeControl(ac_type);
        for (const auto &tt : ac->get_token_types()) {
            token_to_ac_type.insert({tt, ac_type});
        }
    }
    return token_to_ac_type;
}

std::map<midi::TOKEN_TYPE,midi::ATTRIBUTE_CONTROL_TYPE> TOKEN_TO_ATTRIBUTE_CONTROL_TYPE = getTokenToAttributeControlTypeMap();
std::multimap<midi::TOKEN_TYPE,midi::ATTRIBUTE_CONTROL_TYPE> TOKEN_TO_ATTRIBUTE_CONTROL_TYPE_MULTIMAP = getTokenToAttributeControlTypeMultimap();

midi::ATTRIBUTE_CONTROL_TYPE getAttributeControlTypeFromToken(midi::TOKEN_TYPE tt) {
    auto result = TOKEN_TO_ATTRIBUTE_CONTROL_TYPE.find(tt);
    if (result != TOKEN_TO_ATTRIBUTE_CONTROL_TYPE.end()) {
        return result->second;
    }
    return midi::ATTRIBUTE_CONTROL_END;
}

midi::ATTRIBUTE_CONTROL_TYPE getAttributeControlTypeFromTokenMultimap(midi::TOKEN_TYPE tt) {
    auto result = TOKEN_TO_ATTRIBUTE_CONTROL_TYPE_MULTIMAP.find(tt);
    if (result != TOKEN_TO_ATTRIBUTE_CONTROL_TYPE_MULTIMAP.end()) {
        return result->second;
    }
    return midi::ATTRIBUTE_CONTROL_END;
}

// deprecated
int get_token_domain_size(midi::TOKEN_TYPE tt) {
    auto ac_type = getAttributeControlTypeFromTokenMultimap(tt);
    if (ac_type != midi::ATTRIBUTE_CONTROL_END) {
        return getAttributeControl(ac_type)->get_token_domain_size(tt);
    }
    std::cout << "encoder::get_token_domain_size() token type = " << util_protobuf::enum_to_string(tt) << " not found." << std::endl;
    throw std::runtime_error("encoder::get_token_domain_size() token type not found.");
}

// deprecated
std::pair<midi::TOKEN_TYPE,TOKEN_DOMAIN> add_attribute_control_to_representation(midi::TOKEN_TYPE tt) {
    return std::make_pair(tt, TOKEN_DOMAIN(get_token_domain_size(tt)));
}


std::vector<std::pair<midi::TOKEN_TYPE,TOKEN_DOMAIN>> add_attribute_control_to_representation_v2(midi::ATTRIBUTE_CONTROL_TYPE ac_type) {
    std::vector<std::pair<midi::TOKEN_TYPE,TOKEN_DOMAIN>> token_domains;
    auto ac = getAttributeControl(ac_type);
    for (const auto &tt :ac->get_token_types()) {
        token_domains.push_back(std::make_pair(tt, ac->get_token_domain(tt)));
    }
    return token_domains;
}

std::vector<std::tuple<midi::TOKEN_TYPE,int>> get_instrument_exclusive_token_types() {
    std::vector<std::tuple<midi::TOKEN_TYPE,int>> token_types;
    for (const auto &ac : getAttributeControls()) {
        if (ac->track_type == ATTRIBUTE_CONTROL_TRACK_TYPE_INSTRUMENT) {
            if (ac->token_types_v3.size()) {
                for (const auto &tt : ac->token_types_v3) {
                    token_types.push_back(std::make_tuple(std::get<0>(tt),std::get<1>(tt)));
                }
            }
            else {
                for (const auto &tt : ac->get_token_types()) {
                    token_types.push_back(std::make_tuple(tt,0));
                }
            }
        }
    }
    return token_types;
}

std::vector<std::tuple<midi::TOKEN_TYPE,int>> get_drum_exclusive_token_types() {
    std::vector<std::tuple<midi::TOKEN_TYPE,int>> token_types;
    for (const auto &ac : getAttributeControls()) {
        if (ac->track_type == ATTRIBUTE_CONTROL_TRACK_TYPE_DRUM) {
            if (ac->token_types_v3.size()) {
                for (const auto &tt : ac->token_types_v3) {
                    token_types.push_back(std::make_tuple(std::get<0>(tt),std::get<1>(tt)));
                }
            }
            else {
                for (const auto &tt : ac->get_token_types()) {
                    token_types.push_back(std::make_tuple(tt,0));
                }
            }
        }
    }
    return token_types;
}

// refactoring attribute control graph functions
std::vector<midi::TOKEN_TYPE> get_attribute_control_graph(ATTRIBUTE_CONTROL_LEVEL acl, midi::TOKEN_TYPE start, midi::TOKEN_TYPE end) {
    std::vector<midi::TOKEN_TYPE> token_order;
    if (start != midi::TOKEN_NONE) {
        token_order.push_back(start);
    }
    for (const auto &ac : getAttributeControls()) {
        if (ac->control_level == acl) {
            for (const auto &tt : ac->get_token_types()) {
                token_order.push_back(tt);
            }
        }
    }
    if (end != midi::TOKEN_NONE) {
        token_order.push_back(end);
    }
    return token_order;
}

std::vector<midi::TOKEN_TYPE> get_track_pre_instrument_attribute_control_graph() {
    return get_attribute_control_graph(ATTRIBUTE_CONTROL_LEVEL_TRACK_PRE_INSTRUMENT, midi::TOKEN_TRACK, midi::TOKEN_INSTRUMENT);
}

std::vector<midi::TOKEN_TYPE> get_track_attribute_control_graph() {
    return get_attribute_control_graph(ATTRIBUTE_CONTROL_LEVEL_TRACK, midi::TOKEN_INSTRUMENT, midi::TOKEN_BAR);
}

std::vector<midi::TOKEN_TYPE> get_bar_attribute_control_graph() {
    return get_attribute_control_graph(ATTRIBUTE_CONTROL_LEVEL_BAR, midi::TOKEN_BAR, midi::TOKEN_TIME_SIGNATURE);
}


std::vector<std::tuple<midi::TOKEN_TYPE,int>> get_attribute_control_graph_v2(ATTRIBUTE_CONTROL_LEVEL acl, std::tuple<midi::TOKEN_TYPE,int> start, std::tuple<midi::TOKEN_TYPE,int> end) {
    std::vector<std::tuple<midi::TOKEN_TYPE,int>> token_order;
    if (std::get<0>(start) != midi::TOKEN_NONE) {
        token_order.push_back(start);
    }
    for (const auto &ac : getAttributeControls()) {
        if (ac->control_level == acl) {
            if (ac->token_types_v3.size()) {
                for (const auto &x : ac->token_types_v3) {
                    token_order.push_back(std::make_tuple(std::get<0>(x), std::get<1>(x)));
                }
            }
            else {
                for (const auto &tt : ac->get_token_types()) {
                    token_order.push_back(std::make_tuple(tt, 0));
                }
            }
        }
    }
    if (std::get<0>(end) != midi::TOKEN_NONE) {
        token_order.push_back(end);
    }
    return token_order;
}

std::vector<std::tuple<midi::TOKEN_TYPE,int>> get_track_pre_instrument_attribute_control_graph_v2() {
    return get_attribute_control_graph_v2(ATTRIBUTE_CONTROL_LEVEL_TRACK_PRE_INSTRUMENT, std::make_tuple(midi::TOKEN_TRACK, 0), std::make_tuple(midi::TOKEN_INSTRUMENT, 0));
}

std::vector<std::tuple<midi::TOKEN_TYPE,int>> get_track_attribute_control_graph_v2() {
    return get_attribute_control_graph_v2(ATTRIBUTE_CONTROL_LEVEL_TRACK, std::make_tuple(midi::TOKEN_INSTRUMENT, 0), std::make_tuple(midi::TOKEN_BAR, 0));
}

std::vector<std::tuple<midi::TOKEN_TYPE,int>> get_bar_attribute_control_graph_v2() {
    return get_attribute_control_graph_v2(ATTRIBUTE_CONTROL_LEVEL_BAR, std::make_tuple(midi::TOKEN_BAR, 0), std::make_tuple(midi::TOKEN_TIME_SIGNATURE, 0));
}

void override_attribute_controls(const std::shared_ptr<REPRESENTATION> &rep, midi::Piece *x, midi::Status *s) {
    for (const auto &kv : rep->token_domains) {
        auto ac_type = getAttributeControlTypeFromToken(kv.first);
        if (ac_type != midi::ATTRIBUTE_CONTROL_END) {
            getAttributeControl(ac_type)->override_features(x, s);
        }
    }
}

void compute_attribute_controls(const std::shared_ptr<REPRESENTATION> &rep, midi::Piece *x) {
    for (const auto &kv : rep->token_domains) {
        auto ac_type = getAttributeControlTypeFromToken(kv.first);
        if (ac_type != midi::ATTRIBUTE_CONTROL_END) {
            getAttributeControl(ac_type)->compute_features(x);
        }
    }
}

void compute_piece_level_attribute_controls(const std::shared_ptr<REPRESENTATION> &rep, midi::Piece *x) {
    for (const auto &kv : rep->token_domains) {
        auto ac_type = getAttributeControlTypeFromToken(kv.first);
        if (ac_type != midi::ATTRIBUTE_CONTROL_END) {
            auto ac = getAttributeControl(ac_type);
            if ((ac->control_level == ATTRIBUTE_CONTROL_LEVEL_PIECE) || (ac->precompute_on_piece)) {
                ac->compute_piece_level_features(x);
            }
        }
    }
}

std::string compute_all_attribute_controls_py(std::string &piece_json) {
    midi::Piece piece;
    util_protobuf::string_to_protobuf(piece_json, &piece);
    for (const auto &ac : getAttributeControls()) {
        ac->compute_features(&piece);
    }
    return util_protobuf::protobuf_to_string(&piece);
}

void append_track_pre_instrument_tokens(data_structures::TokenSequence *tokens, const std::shared_ptr<REPRESENTATION> &rep, midi::TrackFeatures *tf, bool is_drum) {
    // order of tokens is important here
    for (const auto &tt : getAttributeControlTokenTypes()) {
        if (rep->token_domains.find(tt) != rep->token_domains.end()) {
            auto ac_type = getAttributeControlTypeFromToken(tt);
            if (ac_type != midi::ATTRIBUTE_CONTROL_END) {
                auto ac = getAttributeControl(ac_type);
                if ((ac->control_level == ATTRIBUTE_CONTROL_LEVEL_TRACK_PRE_INSTRUMENT) && (ac->check_valid_track(is_drum))) {
                    ac->append_track_tokens(tokens, rep, tf);
                }
            }
        }   
    }
}

void append_track_tokens(data_structures::TokenSequence *tokens, const std::shared_ptr<REPRESENTATION> &rep, midi::TrackFeatures *tf, bool is_drum) {
    // order of tokens is important here
    for (const auto &tt : getAttributeControlTokenTypes()) {
        if (rep->token_domains.find(tt) != rep->token_domains.end()) {
            auto ac_type = getAttributeControlTypeFromToken(tt);
            if (ac_type != midi::ATTRIBUTE_CONTROL_END) {
                auto ac = getAttributeControl(ac_type);
                if ((ac->control_level == ATTRIBUTE_CONTROL_LEVEL_TRACK) && (ac->check_valid_track(is_drum))) {
                    ac->append_track_tokens(tokens, rep, tf);
                }
            }
        }   
    }
}

void append_bar_tokens(data_structures::TokenSequence *tokens, const std::shared_ptr<REPRESENTATION> &rep, midi::BarFeatures *bf, bool is_drum) {
    // order of tokens is important here
    for (const auto &tt : getAttributeControlTokenTypes()) {
        if (rep->token_domains.find(tt) != rep->token_domains.end()) {
            auto ac_type = getAttributeControlTypeFromToken(tt);
            if (ac_type != midi::ATTRIBUTE_CONTROL_END) {
                auto ac = getAttributeControl(ac_type);
                if ((ac->control_level == ATTRIBUTE_CONTROL_LEVEL_BAR) && (ac->check_valid_track(is_drum))) {
                    ac->append_bar_tokens(tokens, rep, bf);
                }
            }
        }   
    }
}

void set_track_masks(const std::shared_ptr<REPRESENTATION> &rep, std::vector<int> &mask, midi::StatusTrack *track) {
    for (const auto &kv : rep->token_domains) {
        auto ac_type = getAttributeControlTypeFromToken(kv.first);
        if (ac_type != midi::ATTRIBUTE_CONTROL_END) {
            auto ac = getAttributeControl(ac_type);
            if (ac->is_track_control()) {
                ac->set_track_mask(rep, mask, track);
            }
        }
    }
}

void set_bar_masks(const std::shared_ptr<REPRESENTATION> &rep, std::vector<int> &mask, midi::StatusBar *bar) {
    for (const auto &kv : rep->token_domains) {
        auto ac_type = getAttributeControlTypeFromToken(kv.first);
        if (ac_type != midi::ATTRIBUTE_CONTROL_END) {
            auto ac = getAttributeControl(ac_type);
            if (ac->is_bar_control()) {
                ac->set_bar_mask(rep, mask, bar);
            }
        }
    }
}

}
