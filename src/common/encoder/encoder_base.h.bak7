#pragma once

#include <algorithm>

#include "representation.h"
#include "util.h"

#include "../data_structures/encoder_config.h"
#include "../data_structures/train_config.h"
#include "../data_structures/token_sequence.h"
#include "../midi_parsing/midi_io.h"

// START OF NAMESPACE
namespace encoder {

template<class T>
using matrix = std::vector<std::vector<T>>;

std::vector<int> resolve_bar_infill_tokens(std::vector<int> &raw_tokens, const std::shared_ptr<REPRESENTATION> &rep) {
  data_structures::LOGGER(data_structures::VERBOSITY_LEVEL_TRACE, "resolving bar infill" );
  int fill_pholder = rep->encode(midi::TOKEN_FILL_IN_PLACEHOLDER, 0);
  int fill_start = rep->encode(midi::TOKEN_FILL_IN_START, 0);
  int fill_end = rep->encode(midi::TOKEN_FILL_IN_END, 0);

  std::vector<int> tokens;

  auto start_pholder = raw_tokens.begin();
  auto start_fill = raw_tokens.begin();
  auto end_fill = raw_tokens.begin();

  while (start_pholder != raw_tokens.end()) {
    start_pholder = next(start_pholder); // FIRST TOKEN IS PIECE_START ANYWAYS
    auto last_start_pholder = start_pholder;
    start_pholder = find(start_pholder, raw_tokens.end(), fill_pholder);
    if (start_pholder != raw_tokens.end()) {
      start_fill = find(next(start_fill), raw_tokens.end(), fill_start);
      end_fill = find(next(end_fill), raw_tokens.end(), fill_end);

      // insert from last_start_pholder --> start_pholder
      tokens.insert(tokens.end(), last_start_pholder, start_pholder);
      tokens.insert(tokens.end(), next(start_fill), end_fill);
    }
    else {
      // insert from last_start_pholder --> end of sequence (excluding fill)
      start_fill = find(raw_tokens.begin(), raw_tokens.end(), fill_start);
      tokens.insert(tokens.end(), last_start_pholder, start_fill);
    }
  }
  return tokens;
}

class ENCODER {
public:

  virtual ~ENCODER() {}

  // helper for simplicity
  // also used to keep track of attribute controls used ....

  std::vector<std::string> get_attribute_control_types() {
    std::vector<std::string> types;
    auto enum_descriptor = google::protobuf::GetEnumDescriptor<midi::ATTRIBUTE_CONTROL_TYPE>();
    for (auto c : attribute_control_types) {
      types.push_back(enum_descriptor->FindValueByNumber(c)->name());
    }
    return types;
  }

  virtual void preprocess_piece(midi::Piece *p) {
    // default is to do nothing
  }

  std::vector<int> encode(midi::Piece *p) {
    preprocess_piece(p);
    data_structures::TokenSequence ts = encode_piece(p);
    return ts.tokens;
  }

  std::vector<int> encode_wo_preprocess(midi::Piece *p) {
    data_structures::TokenSequence ts = encode_piece(p);
    return ts.tokens;
  }

  virtual void decode(std::vector<int> &tokens, midi::Piece *p) {
    if (config->do_multi_fill == true) {
      tokens = resolve_bar_infill_tokens(tokens, rep);
    }
    data_structures::LOGGER(data_structures::VERBOSITY_LEVEL_TRACE, "AFTER BAR INFILL RESOLVED :: ");
    for (int tok : tokens) {
      data_structures::LOGGER(data_structures::VERBOSITY_LEVEL_TRACE, pretty(tok));
    }
    decode_track(tokens, p, rep, config);
  }

  std::string midi_to_json(const std::string &filepath) {
    midi::Piece p;
    midi_io::ParseSong(filepath, &p, config);
    preprocess_piece(&p); // add features that the encoder may need
    std::string json_string;
    google::protobuf::util::MessageToJsonString(p, &json_string);
    return json_string;
  }

  void midi_to_piece(const std::string& filepath, midi::Piece* p) {
    midi_io::ParseSong(filepath, p, config);
    preprocess_piece(p);
  }
  
  std::vector<int> midi_to_tokens(std::string &filepath) {
    midi::Piece p;
    midi_io::ParseSong(filepath, &p, config);
    data_structures::LOGGER(data_structures::VERBOSITY_LEVEL_TRACE, data_structures::to_str("Parsed File :: ",util_protobuf::protobuf_to_string(&p)));
    return encode(&p);
  }

  void json_to_midi(std::string &json_string, std::string &filepath) {
    midi::Piece p;
    google::protobuf::util::JsonStringToMessage(json_string.c_str(), &p);
    midi_io::write_midi(&p, filepath, -1);
  }

  std::string json_to_json(std::string &json_string_in) {
    midi::Piece p;
    google::protobuf::util::JsonStringToMessage(json_string_in.c_str(), &p);
    std::string json_string;
    google::protobuf::util::MessageToJsonString(p, &json_string);
    return json_string;
  }

  void json_track_to_midi(std::string &json_string, std::string &filepath, int single_track) {
    midi::Piece p;
    google::protobuf::util::JsonStringToMessage(json_string.c_str(), &p);
    midi_io::write_midi(&p, filepath, single_track);
  }

  std::vector<int> json_to_tokens(std::string &json_string) {
    midi::Piece p;
    google::protobuf::util::JsonStringToMessage(json_string.c_str(), &p);
    return encode(&p); 
  }

  std::string tokens_to_json(std::vector<int> &tokens) {
    midi::Piece p;
    decode(tokens, &p);
    std::string json_string;
    google::protobuf::util::MessageToJsonString(p, &json_string);
    data_structures::LOGGER(data_structures::VERBOSITY_LEVEL_TRACE, data_structures::to_str("Decoded File :: ",json_string));
    return json_string;
  }

  void resample_delta(midi::Piece *p) {
    // This function rewrites the piece events time values to take in account their delta values
    data_structures::LOGGER(data_structures::VERBOSITY_LEVEL_VERBOSE, "Resampling Piece with Delta values");

    //We have to deal with overlapping notes by applying next notes onset delta to previous notes offset
    std::map<int, int> delta_to_apply;
    int track_num = 0;
    for (const auto &track : p->tracks()) {
      int bar_num = 0;
      for (const auto &bar : track.bars()) {
        std::map<int, std::vector<std::tuple<int, int, int, int>>> pitch_to_events;
        for (int i=0; i<bar.events_size(); i++) {
          int event_idx = bar.events()[i];
          midi::Event event = p->events(event_idx);
          pitch_to_events[event.pitch()].push_back(std::make_tuple(event_idx, event.time(), event.velocity(), event.delta()));
        }
        for (auto line : pitch_to_events) {
          std::sort(line.second.begin(), line.second.end(), [](std::tuple<int, int, int, int> a, std::tuple<int, int, int, int> b) {
            if (std::get<1>(a) < std::get<1>(b)) return true;
            if (std::get<1>(b) < std::get<1>(a)) return false;
            return (std::get<2>(a) < std::get<2>(b));
          });
          std::tuple<int, int, int> last_event;
          int last_offset_idx = -1;
          for (auto const& e : line.second) {
            // if onset, check last offset
            if ((std::get<2>(e) > 0) && (last_offset_idx != -1)) {
              if ((std::get<3>(e) != 0) && (std::get<1>(e) == p->events(last_offset_idx).time())) {
                delta_to_apply[last_offset_idx] = std::get<3>(e);
              }
            } else if (std::get<2>(e) == 0) {
              last_offset_idx = std::get<0>(e);
            }
          }
        }
        bar_num++;
      }
      track_num++;
    }

    int current_res = config->resolution;
    int target_res = config->decode_resolution;
    p->set_resolution(target_res);
    p->set_internal_ticks_per_quarter(target_res);
    int old_time, new_time, delta;
    std::vector<std::tuple<int, midi::Event>> events_cache;
    // Get all events and store in cache vector

    int num_events = p->events_size();
    for (int event_index=0; event_index<num_events; event_index++) {
      midi::Event e = p->events(event_index);
      old_time = e.time();
      delta = e.delta();
      if (delta_to_apply.count(event_index) > 0) {
        assert(delta_to_apply.count(event_index) == 1);
        delta = delta_to_apply[event_index]; 
      }
      // We round down to be safe
      new_time = (int)(target_res * old_time / current_res);
      //exclude negative times
      new_time = std::max(new_time + delta, 0);
      // Set new resampled time
      e.set_time(new_time);
      events_cache.push_back(std::make_tuple(event_index, e));
    }
    // Sort events to replace in the correct order
    sort(events_cache.begin(), events_cache.end(), [](std::tuple<int, midi::Event> a, std::tuple<int, midi::Event> b) { 
        return std::get<0>(a) < std::get<0>(b); 
      });
    // Clear all events now that they're cached
    p->clear_events();
    // Reinject resampled events 
    for (const std::tuple<int, midi::Event> &oe : events_cache) {
      midi::Event *ne = p->add_events();
      ne->CopyFrom( std::get<1>(oe) );
    }
    assert(num_events == p->events_size());
  }

  std::string resample_delta_json(std::string &json_string) {
    std::string res_json_string;
    midi::Piece p;
    google::protobuf::util::JsonStringToMessage(json_string.c_str(), &p);
    if (config->use_microtiming) {
      resample_delta(&p);
    }
    google::protobuf::util::MessageToJsonString(p, &res_json_string);
    return res_json_string;
  }

  void tokens_to_json_array(std::vector<std::vector<int>> &seqs, std::vector<midi::Piece> &output) {
    for (int i=0; i<(int)seqs.size(); i++) {
      decode(seqs[i], &(output[i]));
    }
  }

  void tokens_to_midi(std::vector<int> &tokens, std::string &filepath) {
    midi::Piece p;
    decode(tokens, &p);
    midi_io::write_midi(&p, filepath, -1);
  }

  // ====================
  // expose methods of rep that we need

  std::string pretty(int token) {
    return rep->pretty(token);
  }

  int vocab_size() {
    return rep->vocab_size;
  }

  // ====================

  // below is a simplified refactor of the encoding process
  // broken into clear functions to
  // - encode notes within a bar
  // - encode a bar
  // - encode a track
  // - encode a piece

  // ====================

  void encode_notes(int bar_num, int track_num, midi::Piece *p, data_structures::TokenSequence *ts) {
    const auto track = p->tracks(track_num);
    const auto bar = track.bars(bar_num);
    const auto is_drum = data_structures::is_drum_track(track.track_type());
    const int N_DURATION_TOKENS = rep->get_domain_size(midi::TOKEN_NOTE_DURATION);
    int N_TIME_TOKENS = rep->get_domain_size(midi::TOKEN_DELTA);

    // group notes by onset time
    std::vector<int> onsets;
    std::vector<int> onsets_idx;
    std::map<int, std::vector<int>> notes_by_onset;
    std::map<int, int> delta_onsets;
    int idx = 0;
    for (const auto &i : bar.events()) {
      midi::Event event = p->events(i);
      if ((event.internal_duration() > 0) && (event.velocity() > 0)) {
        if (notes_by_onset.find(event.time()) == notes_by_onset.end()) {
          onsets.push_back(event.time());
          onsets_idx.push_back(idx);
          idx += 1;
        }
        notes_by_onset[event.time()].push_back(i);
        delta_onsets[i] = event.delta();
      }
    }

    int last_velocity = -1;
    int onset;
    int d_onset;
    for (const auto &idx : onsets_idx) {
      onset = onsets[idx];
      // checking for onset > 0 is to make things backwards compatible with the old representation
      // however for randomly ordering onset times we need to include onset == 0
      if ((onset > 0)) { 
        ts->push_back( rep->encode(midi::TOKEN_TIME_ABSOLUTE_POS, onset) );
      }
      
      for (const auto &i : notes_by_onset[onset]) {
        midi::Event event = p->events(i);
        d_onset = delta_onsets[i];
        if (rep->has_token_type(midi::TOKEN_VELOCITY_LEVEL)) {
          int current_velocity = rep->encode_partial(midi::TOKEN_VELOCITY_LEVEL, event.velocity());
          if ((current_velocity > 0) && (current_velocity != last_velocity)) {
            ts->push_back( rep->encode(midi::TOKEN_VELOCITY_LEVEL, event.velocity()) );
            last_velocity = current_velocity;
          }
        }
        if (config->use_microtiming) {
          if (d_onset < 0) {
            ts->push_back( rep->encode(midi::TOKEN_DELTA_DIRECTION, 0) );
            d_onset *= -1;
          }
          d_onset = std::min(N_TIME_TOKENS - 1, d_onset);
          if (d_onset > 0) {
            ts->push_back( rep->encode(midi::TOKEN_DELTA, d_onset) );
          }
        }
        ts->push_back( rep->encode(midi::TOKEN_NOTE_ONSET, event.pitch()) );
        if (!is_drum) {
          ts->push_back( rep->encode(midi::TOKEN_NOTE_DURATION, std::min(event.internal_duration(), N_DURATION_TOKENS)-1) );
        }
      }
    }
  }

  void encode_bar(int bar_num, int track_num, midi::Piece *p, data_structures::TokenSequence *ts, bool infill) {
    auto track = p->tracks(track_num);
    const auto bar = track.bars(bar_num);
    const auto is_drum = data_structures::is_drum_track(track.track_type());

    ts->on_bar_start(p, rep);

    if (infill) {
      ts->push_back( rep->encode(midi::TOKEN_FILL_IN_START, 0) );
      encode_notes(bar_num, track_num, p, ts);
      ts->push_back( rep->encode(midi::TOKEN_FILL_IN_END, 0) );
    }
    else {
      ts->push_back( rep->encode(midi::TOKEN_BAR, 0) );

      midi::BarFeatures *bf = util_protobuf::GetBarFeatures(&track, bar_num);
      append_bar_tokens(ts, rep, bf, is_drum);

      if (rep->has_token_type(midi::TOKEN_TIME_SIGNATURE)) {
        ts->push_back( rep->encode(midi::TOKEN_TIME_SIGNATURE, std::make_tuple(bar.ts_numerator(), bar.ts_denominator())) );
      }

      if ((config->do_multi_fill) && (config->multi_fill.find(std::make_pair(track_num,bar_num)) != config->multi_fill.end())) {
        ts->push_back( rep->encode(midi::TOKEN_FILL_IN_PLACEHOLDER, 0) );
      }
      else {
        encode_notes(bar_num, track_num, p, ts);
      }
      ts->push_back( rep->encode(midi::TOKEN_BAR_END, 0) );
    }
  }

  void encode_track(int track_num, midi::Piece *p, data_structures::TokenSequence *ts) {
    const auto track = p->tracks(track_num);
    const auto is_drum = data_structures::is_drum_track(track.track_type());
    const auto f = util_protobuf::GetTrackFeatures(p, track_num);

    ts->on_track_start(p, rep);

    ts->push_back( rep->encode(midi::TOKEN_TRACK, track.track_type()) );

    append_track_pre_instrument_tokens(ts, rep, f, is_drum);

    if (rep->has_token_type(midi::TOKEN_INSTRUMENT)) {
      int inst = track.instrument();
      ts->push_back( rep->encode(midi::TOKEN_INSTRUMENT, inst) );
    }

    append_track_tokens(ts, rep, f, is_drum);

    for (int i=0; i<track.bars_size(); i++) {
      encode_bar(i, track_num, p, ts, false);
    }

    ts->push_back( rep->encode(midi::TOKEN_TRACK_END, 0) );
  }

  data_structures::TokenSequence encode_piece(midi::Piece *p) {

    // make sure that rep does not try use deprecated note encodings
    if ((!rep->has_token_type(midi::TOKEN_NOTE_DURATION)) || (!rep->has_token_type(midi::TOKEN_TIME_ABSOLUTE_POS))) {
      throw std::runtime_error("ERROR: ENCODING PIECE WITH DEPRECATED NOTE ENCODINGS");
    }

    data_structures::TokenSequence ts(rep);

    ts.push_back( rep->encode(
      midi::TOKEN_PIECE_START, std::min((int)config->do_multi_fill,rep->get_domain_size(midi::TOKEN_PIECE_START)-1)));

    if (rep->has_token_type(midi::TOKEN_NUM_BARS)) {
      ts.push_back( rep->encode(midi::TOKEN_NUM_BARS, util_protobuf::GetNumBars(p)) );
    }

    for (int i=0; i<p->tracks_size(); i++) {
      encode_track(i, p, &ts);
    }

    if (config->do_multi_fill) {
      for (const auto &track_bar : config->multi_fill) {      
        encode_bar(std::get<1>(track_bar), std::get<0>(track_bar), p, &ts, true);
      }
    }

    return ts;
  }

  std::shared_ptr<REPRESENTATION> get_rep() {
    return rep;
  }

  std::shared_ptr<data_structures::EncoderConfig> config;
  std::shared_ptr<REPRESENTATION> rep;
  std::vector<midi::ATTRIBUTE_CONTROL_TYPE> attribute_control_types;
};

}
// END OF NAMESPACE
