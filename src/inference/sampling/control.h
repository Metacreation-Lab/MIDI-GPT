// this code tracks progress

// constrain bars per track
// constrain timesteps per bar
// constrain max polyphony
// constrain offsets to notes that have been onset

#pragma once

#include <map>
#include <tuple>
#include <set>
#include <vector>

#include "../../common/encoder/representation.h"
#include "../../common/encoder/encoder_base.h"
#include "../enum/encoder_types.h"
#include "../../common/encoder/attribute_control.h"
#include "../../common/data_structures/verbosity.h"
#include "graph.h"

namespace sampling {

using TOKEN_EDGE = std::pair<midi::TOKEN_TYPE,midi::TOKEN_TYPE>;
using CKPT_MAP_TYPE = std::map<std::tuple<int,enums::MODEL_TYPE>,std::tuple<enums::ENCODER_TYPE,std::string>>;

class CONDITIONAL_REP_GRAPH {
public:

  virtual ~CONDITIONAL_REP_GRAPH() {}

  virtual bool is_active(midi::TRACK_TYPE track_type) {
    throw std::runtime_error("CONDITIONAL_REP_GRAPH::is_active() : NOT IMPLEMENTED");
  }

  template <typename T>
  void show(const std::vector<T> &x, const std::string &s) {
    std::ostringstream buffer;
    buffer << s << " :: [";
    for (const auto &e : x) {
      buffer << util_protobuf::enum_to_string(e) << ", ";
    }
    buffer << "]";
    data_structures::LOGGER(buffer.str());
  }

  midi::TOKEN_TYPE possibly_skip(midi::TRACK_TYPE track_type, int last_token, const std::unique_ptr<REP_GRAPH> &rg, std::shared_ptr<encoder::REPRESENTATION> &rep, std::vector<int> & mask) {
    
    auto target_node = std::make_tuple(midi::TOKEN_NONE, 0);

    if (is_active(track_type)) {
      auto inferred_node = rg->graph.infer_node(last_token, rg->enc);
      auto next_nodes = graph->graph.get_next_nodes(inferred_node);
      auto next_global_nodes = rg->graph.get_next_nodes(inferred_node);

      if ((next_nodes.size() == 1)) {
        int loop_count = 0;
        while ((next_global_nodes.size() == 1) && (next_nodes[0] != next_global_nodes[0])) {
          target_node = next_global_nodes[0];
          next_global_nodes = rg->graph.get_next_nodes(target_node);
          loop_count++;

          if (loop_count > 100) {
            throw std::runtime_error("CONDITIONAL_REP_GRAPH::possibly_skip() : INFINITE LOOP");
          }
        }

        if (std::get<0>(target_node) != midi::TOKEN_NONE) {
          auto msg = data_structures::to_str("CONDITIONAL_REP_GRAPH::possibly_skip() : skip ", util_protobuf::enum_to_string(std::get<0>(target_node)), std::get<1>(target_node));
          std::cout << msg << std::endl;
          data_structures::LOGGER(msg);
          rg->graph.skip(rg->graph.get_previous_nodes(target_node)[0]);
          rg->set_mask(rep->encode(std::get<0>(target_node), 0), mask);

        }

      }
    }
    return std::get<0>(target_node);
  }

  std::unique_ptr<REP_GRAPH> graph;

};


class INSTRUMENT_CONDITIONAL_REP_GRAPH : public CONDITIONAL_REP_GRAPH {
public:

  INSTRUMENT_CONDITIONAL_REP_GRAPH(encoder::ENCODER *e, enums::MODEL_TYPE mt) {
    graph = std::make_unique<REP_GRAPH>(e,mt,encoder::get_drum_exclusive_token_types());
    data_structures::LOGGER(data_structures::VERBOSITY_LEVEL_TRACE, "INSTRUMENT_CONDITIONAL_REP_GRAPH" );
  }

  bool is_active(midi::TRACK_TYPE track_type) {
    return (data_structures::is_drum_track(track_type) == false);
  }

};


class DRUM_CONDITIONAL_REP_GRAPH : public CONDITIONAL_REP_GRAPH {
public:

  DRUM_CONDITIONAL_REP_GRAPH(encoder::ENCODER *e, enums::MODEL_TYPE mt) {
    graph = std::make_unique<REP_GRAPH>(e,mt,encoder::get_instrument_exclusive_token_types());
    data_structures::LOGGER(data_structures::VERBOSITY_LEVEL_TRACE, "DRUM_CONDITIONAL_REP_GRAPH" );
  }

  bool is_active(midi::TRACK_TYPE track_type) {
    return data_structures::is_drum_track(track_type);
  }

};


class SAMPLE_CONTROL {
public:
  SAMPLE_CONTROL(midi::Piece *piece, midi::Status *status, midi::HyperParam *param, midi::ModelMetadata *meta) {   
    data_structures::LOGGER(data_structures::VERBOSITY_LEVEL_TRACE, "SAMPLE_CONTROL" );

    verbose = param->verbose();

    data_structures::LOGGER(data_structures::VERBOSITY_LEVEL_TRACE, util_protobuf::protobuf_to_string(status));
    
    initialize(piece, status, param, meta);
    rep = enc->rep;
    rg = std::make_unique<REP_GRAPH>(enc.get(), model_type);
    instrument_rg = std::make_unique<INSTRUMENT_CONDITIONAL_REP_GRAPH>(enc.get(), model_type);
    drum_rg = std::make_unique<DRUM_CONDITIONAL_REP_GRAPH>(enc.get(), model_type);

    if ((!rg) || (!instrument_rg) || (!drum_rg)) {
      std::runtime_error("REP GRAPH CONSTRUCTOR FAILED");
    }
    else {
      data_structures::LOGGER( "REP GRAPH CONSTRUCTOR SUCCESS" );
    }

    parse_status(status);
    initialize_members();

  }

  ~SAMPLE_CONTROL() {}

  void initialize_members() {
      data_structures::LOGGER(data_structures::VERBOSITY_LEVEL_TRACE, "initialize_members" );
    barlength = 4 * enc->config->resolution;
    timestep = 0;
    absolute_timestep = 0;
    bar_count = 0;
    track_count = 0;
    infill_bar_count = 0;
    finished = false;
    token_position = 0;
    num_delta_tokens = 0;
  }

  void set_bar_infill_prompt(std::vector<std::tuple<int,int>> &bars, midi::Piece *p, midi::Status *status, midi::HyperParam *param) {
      data_structures::LOGGER(data_structures::VERBOSITY_LEVEL_TRACE, "set_bar_infill_prompt" );

    if (p) {
      std::set<std::tuple<int,int>> barset;
      std::copy(bars.begin(), bars.end(), std::inserter(barset, barset.end()));
      enc->config->do_multi_fill = true;
      enc->config->multi_fill = barset;
      
      if (param->internal_skip_preprocess()) {
        util_protobuf::calculate_note_durations(p);
        prompt = enc->encode_wo_preprocess(p);
      }
      else {
        prompt = enc->encode(p);
      }
      
      data_structures::LOGGER( "FULL PROMPT " );
      for (int i=0; i<(int)prompt.size(); i++) {
        data_structures::LOGGER( enc->rep->pretty(prompt[i]) );
      }
      data_structures::LOGGER( "FULL PROMPT " );
      
      int fill_start = enc->rep->encode(midi::TOKEN_FILL_IN_START,0);
      for (int index=0; index<(int)prompt.size(); index++) {
        if (prompt[index] == fill_start) {
          prompt.resize(index+1);
          break;
        }
      }
    }
    else {
      throw std::runtime_error("MUST PROVIDE midi::Piece FOR BAR INFILL MODE");
    }
  }

  void set_autoregressive_prompt(std::vector<midi::StatusTrack> &tracks, midi::Piece *p, midi::Status *status, midi::HyperParam *param) {
	  data_structures::LOGGER(data_structures::VERBOSITY_LEVEL_TRACE, "set_autoregressive_prompt" );

    enc->config->do_multi_fill = false;

    if (p->tracks_size()) {
      data_structures::LOGGER(data_structures::VERBOSITY_LEVEL_TRACE, "SET AUTOREGRESSIVE PROMPT" );
      prompt = enc->encode(p);
    }
    else {
      prompt.push_back( enc->rep->encode(midi::TOKEN_PIECE_START,0) );
    }
  }

  void initialize(midi::Piece *piece, midi::Status *status, midi::HyperParam *param, midi::ModelMetadata *meta) {
	  data_structures::LOGGER(data_structures::VERBOSITY_LEVEL_TRACE, "initialize" );

    util_protobuf::UpdateHasNotes(piece);

    std::vector<midi::StatusTrack> tracks;
    std::vector<std::tuple<int,int>> bars;
    int num_cond_tracks = 0;
    int num_resample_tracks = 0;
    int num_infill_tracks = 0;
    std::vector<util_protobuf::STATUS_TRACK_TYPE> track_types;
    std::vector<int> order;
    std::vector<int> cond_tracks;

    int track_num = 0;
    for (const auto &track : status->tracks()) {
      util_protobuf::STATUS_TRACK_TYPE tt = util_protobuf::infer_track_type(track);
      data_structures::LOGGER(data_structures::VERBOSITY_LEVEL_TRACE, data_structures::to_str("STATUS TRACK TYPE FOR ",track.track_id()," : ", tt));
      switch( tt ) {
        case util_protobuf::CONDITION:
          order.push_back( num_cond_tracks );
          cond_tracks.push_back( track.track_id() );
          num_cond_tracks++;
          break;
        case util_protobuf::RESAMPLE:
          order.push_back( num_resample_tracks );
          tracks.push_back( track );
          num_resample_tracks++;
          break;
        case util_protobuf::INFILL :     
          num_infill_tracks++;
          break;
      }
      track_types.push_back( tt );
      int bar_num = 0;
      for (const auto &selected : track.selected_bars()) {
        if (selected) {
          bars.push_back( std::make_pair(track_num, bar_num) );
        }
        bar_num++;
      }
      track_num++;
    }

    // provide overview of tracks for sampling
    int verbose_track_num = 0;
    for (const auto &track_type : track_types) {
      data_structures::LOGGER(data_structures::to_str("TRACK ", verbose_track_num, " -> ", track_type));
      verbose_track_num++;
    }

    // select the correct model
    int nb = status->tracks(0).selected_bars_size();

    enc = enums::getEncoderFromString(meta->encoder());

    if (num_infill_tracks > 0) {
      data_structures::LOGGER( "INFILL" );
      model_type = enums::BAR_INFILL_MODEL;

      // remove excess bars if any
      util_protobuf::prune_tracks(
        piece, arange(0,piece->tracks_size(),1), arange(0,nb,1));

      // here track ordering are preserved
      inverse_order = arange(piece->tracks_size());
      data_structures::LOGGER(data_structures::to_str("GENERATING ", bars.size(), " BARS"));
      set_bar_infill_prompt(bars, piece, status, param);

    }
    else {
      data_structures::LOGGER( "TRACK" );
      model_type = enums::TRACK_MODEL;

      data_structures::LOGGER(data_structures::to_str("GENERATING ", num_resample_tracks, " TRACKS"));

      // fix the order
      // order is the output position for each track
      for (track_num=0; track_num<status->tracks_size(); track_num++) {
        if (track_types[track_num] == util_protobuf::RESAMPLE) {
          order[track_num] = order[track_num] + num_cond_tracks;
        }
      }
      inverse_order.resize(order.size());
      for (int i=0; i<(int)order.size(); i++) {
        inverse_order[order[i]] = i;
      }


      // prune unneeded tracks
      util_protobuf::prune_tracks(piece, cond_tracks, arange(0,nb,1));
      
      data_structures::LOGGER( "AFTER PRUNE TRACKS ...." );
      util_protobuf::print_piece_summary(piece);
      data_structures::LOGGER( "============================" );

      set_autoregressive_prompt(tracks, piece, status, param);

    }
  }

  void finalize(midi::Piece *piece) {
    data_structures::LOGGER(data_structures::VERBOSITY_LEVEL_TRACE, "finalize" );
    if (model_type == enums::TRACK_MODEL) {
      data_structures::LOGGER(data_structures::VERBOSITY_LEVEL_TRACE, "Reordering tracks" );
      util_protobuf::reorder_tracks(piece, inverse_order);
    }
  }

  void parse_status(midi::Status *status) {
    data_structures::LOGGER(data_structures::VERBOSITY_LEVEL_TRACE, "parse_status" );

    // for bar-infilling we have to determine one thing
    // 1) the number of bars to be infilled
    int tnum = 0;
    num_infill_bars = 0;
    for (const auto &track : status->tracks()) {
      int bnum = 0;
      for (const auto &bar : track.selected_bars()) {
        // keep track of selected bars
        selected_bars.push_back( std::pair(tnum,bnum) );
        num_infill_bars += (int)bar;
        bnum++;
      }
      tnum++;
    }

    // for track generation we have to determine two things
    // 1) the number of bars per track
    // 2) the token restrictions for attribute control
    //    for this we can use a static mask for each track
    //    as there should be no overlap
    num_bars = status->tracks(0).selected_bars_size();
    num_tracks = status->tracks_size();
    
    for (int i=0; i<status->tracks_size(); i++) {
      midi::StatusTrack track = status->tracks(inverse_order[i]);
      std::vector<int> mask(rep->max_token(),0);

      // add polyphony hard limit
      data_structures::LOGGER(data_structures::VERBOSITY_LEVEL_TRACE, data_structures::to_str("TRACK: ", i, " - POLYPHONY HARD LIMIT: ", track.polyphony_hard_limit()));
      polyphony_hard_limits.push_back( track.polyphony_hard_limit() );

      // add per-track temperature
      track_temperatures.push_back( track.temperature() );

      // num bars
      rep->set_mask(midi::TOKEN_NUM_BARS, {track.selected_bars_size()}, mask, 1);
      
      // track type
      int tt = track.track_type();
      if (tt == midi::STANDARD_BOTH) {
        rep->set_mask(midi::TOKEN_TRACK, {-1}, mask, 1);
      }
      else {
        rep->set_mask(midi::TOKEN_TRACK, {tt}, mask, 1);
      }

      // instrument
      std::vector<int> insts = enums::GM_MOD[track.instrument()];
      if ((enc->rep->has_pretrain_instrument_mapping()) && (tt == midi::STANDARD_TRACK)) {
        // on drum tracks we don't map instruments
        for (int i=0; i<(int)insts.size(); i++) {
          auto it = enums::PRETRAIN_GROUPING_V2.find(insts[i]);
          if (it != enums::PRETRAIN_GROUPING_V2.end()) {
            insts[i] = it->second;
          }
          else {
            throw std::runtime_error("CAN NOT FIND INSTRUMENT IN PRETRAIN GROUPING");
          }
        }
      }
      rep->set_mask(midi::TOKEN_INSTRUMENT, insts, mask, 1);

      // density level
      rep->set_mask(midi::TOKEN_DENSITY_LEVEL, {track.density()-1}, mask, 1);
      
      // min-max polyphony
      rep->set_mask(midi::TOKEN_MIN_POLYPHONY, {track.min_polyphony_q()-1}, mask, 1);
      rep->set_mask(midi::TOKEN_MAX_POLYPHONY, {track.max_polyphony_q()-1}, mask, 1);

      // min-max duration
      rep->set_mask(
          midi::TOKEN_MIN_NOTE_DURATION, {track.min_note_duration_q()-1}, mask, 1);
      rep->set_mask(
          midi::TOKEN_MAX_NOTE_DURATION, {track.max_note_duration_q()-1}, mask, 1);
        
      set_track_masks(rep, mask, &track);

      std::set<midi::TOKEN_TYPE> fixed = {
        midi::TOKEN_NUM_BARS,
        midi::TOKEN_TRACK,
        midi::TOKEN_GENRE,
        midi::TOKEN_INSTRUMENT,
        midi::TOKEN_TIME_SIGNATURE,

        midi::TOKEN_DENSITY_LEVEL,
        midi::TOKEN_MIN_POLYPHONY,
        midi::TOKEN_MAX_POLYPHONY,
        midi::TOKEN_MIN_NOTE_DURATION,
        midi::TOKEN_MAX_NOTE_DURATION,
        
      };

      auto track_control_tokens = encoder::get_track_attribute_control_graph();
      for (int i=0; i<(int)track_control_tokens.size()-1; i++) {
        if (fixed.find(track_control_tokens[i]) == fixed.end()) {
          fixed.insert(track_control_tokens[i]);
        }
      }
      
      if (verbose) {
        // show the attribute mask
        data_structures::LOGGER( "=======================" );
        data_structures::LOGGER( "ATTRIBUTE MASK : " );
        for (int i=0; i<(int)mask.size(); i++) {
          if (mask[i]) {
            data_structures::LOGGER( rep->pretty(i) );
          }
        }
        data_structures::LOGGER( "=======================" );
      }

      for (const auto &kv : rep->token_domains) {
        if (fixed.find(kv.first) == fixed.end()) {
          rep->set_mask(kv.first, {-1}, mask, 1);
        }
      }

      attribute_masks.push_back( mask );

      // changing to use status bar instead
      std::vector<std::vector<int>> bar_masks;
      for (int bn=0; bn<track.bars_size(); bn++) {
        std::vector<int> bar_mask(mask);
        midi::StatusBar bar = track.bars(bn);
        if (rep->has_token_type(midi::TOKEN_TIME_SIGNATURE)) {
          int tstoken = rep->encode(midi::TOKEN_TIME_SIGNATURE, std::make_tuple(bar.ts_numerator(), bar.ts_denominator()));
          bar_mask[tstoken] = 1; // only allow time signature
        }
        set_bar_masks(rep, bar_mask, &bar);
        bar_masks.push_back( bar_mask );
      }
      attribute_bar_masks.push_back( bar_masks );
    
    }
  }

  void update(int token) {
      data_structures::LOGGER(data_structures::VERBOSITY_LEVEL_TRACE, "controlhSAMPLECONTROL update" );
    midi::TOKEN_TYPE tt = rep->get_token_type(token);
    switch (tt) {
      case midi::TOKEN_TRACK: {
        bar_count = 0;
        absolute_timestep = 0;
        bar_start_timestep = 0;
        onsets.clear();
        note_expiry.clear();
        current_track_type = static_cast<midi::TRACK_TYPE>(rep->decode(token));
        break;
      }
      case midi::TOKEN_TRACK_END: {
        track_count += 1;
        break;
      }
      case midi::TOKEN_BAR: {
        timestep = 0;
        barlength = 4 * enc->config->resolution;
        absolute_timestep = bar_start_timestep;
        break;
      }
      case midi::TOKEN_BAR_END: {
        bar_count += 1;
        bar_start_timestep += barlength;
        break;
      }
      case midi::TOKEN_FILL_IN_START: {
        // clear onsets and read in the token sequence
        // we backfill events in between FILL_IN_PLACEHOLDERs
        // so that we don't skip context
        int fillp_token = enc->rep->encode(midi::TOKEN_FILL_IN_PLACEHOLDER,0);
        auto it = history.begin();
        auto prev = it;
        for (int i=0; i<=infill_bar_count; i++) {
          prev = it;
          it = find(next(it), history.end(), fillp_token);
        }
        for (auto i=next(prev); i!=it; i++) {
          if (verbose) {
            data_structures::LOGGER(data_structures::to_str("BACKFILLING :: ", enc->rep->pretty(*i)));
          }
          update(*i);
        }
        break;
      }
      case midi::TOKEN_FILL_IN_END: {
        infill_bar_count += 1;
        break;
      }
      case midi::TOKEN_TIME_SIGNATURE: {
        std::tuple<int,int> ts = rep->decode_timesig(token);
        double ts_ratio = ((double)std::get<0>(ts) / std::get<1>(ts));
        barlength = ts_ratio * 4 * enc->config->resolution;
        break;
      }
      case midi::TOKEN_TIME_ABSOLUTE_POS: {
        int t = rep->decode(token);
        timestep = t;
        absolute_timestep = bar_start_timestep + t;
        break;
      }
      case midi::TOKEN_DELTA: {
        break;
      }
      case midi::TOKEN_NOTE_ONSET: {
        int pitch = rep->decode(token);
        onsets.insert( pitch );

        if (data_structures::is_drum_track(current_track_type)) {
          // artificially add the note duration of 1
          last_token = token;
          update(rep->encode(midi::TOKEN_NOTE_DURATION,0));
        }

        break;
      }
      case midi::TOKEN_NOTE_DURATION: {
        int dur = rep->decode(token) + 1;
        int pitch = rep->decode(last_token);
        note_expiry[dur + absolute_timestep].push_back(pitch);
        break;
      }
      default:
        break;
    }

    // remove notes that have "expired"
    std::vector<int> to_remove;
    for (const auto &kv : note_expiry) {
      if (kv.first <= absolute_timestep) {
        for (const auto &pitch : kv.second) {
          onsets.erase( pitch );
        }
        to_remove.push_back( kv.first );
      }
    }
    // remove these lists from note expiry
    for (const auto &t : to_remove) {
      note_expiry.erase( t ); 
    }

    last_token = token;

    if (verbose) {
      data_structures::LOGGER(data_structures::to_str("ONSETS : ", onsets.size()));
    }
    

  }

  void set_mask(int last_token, std::vector<int> &mask) {
    data_structures::LOGGER(data_structures::VERBOSITY_LEVEL_TRACE, "controlhSAMPLECONTROL set_mask" );

    // basic constraints of the representation    
    midi::TOKEN_TYPE last_tt = rep->get_token_type(last_token);
    bool is_drum = data_structures::is_drum_track(current_track_type);

    // automatically handle skipping tokens when necessary for drum or instrument tracks
    midi::TOKEN_TYPE inst_skip = instrument_rg->possibly_skip(current_track_type, last_token, rg, rep, mask);
    midi::TOKEN_TYPE drum_skip = drum_rg->possibly_skip(current_track_type, last_token, rg, rep, mask);

    if ((is_drum) && (last_tt == midi::TOKEN_NOTE_ONSET)) {
      // fast forward past NOTE_DURATION token
      rg->skip(midi::TOKEN_NOTE_ONSET);
      rg->set_mask(rep->encode(midi::TOKEN_NOTE_DURATION,0), mask);
    }
    else if ((inst_skip == midi::TOKEN_NONE) && (drum_skip == midi::TOKEN_NONE)) {
      rg->set_mask(last_token, mask);
    }

    // can't have onset for note that is already sounding
    for (const auto &pitch : onsets) {
      mask[rep->encode(midi::TOKEN_NOTE_ONSET,pitch)] = 0;
    }    
   
    // can't have note onsets when timestep == barlength
    // you can only have note offsets
    if (timestep == barlength) {
      if (verbose) {
        data_structures::LOGGER( "HIT TIME LIMIT >>>>>>>>>>>>>>>>>>>> " );
      }
      rep->set_mask(midi::TOKEN_NOTE_ONSET, {-1}, mask, 0);
      rep->set_mask(midi::TOKEN_VELOCITY_LEVEL, {-1}, mask, 0);
    }

    // determine what the hard limit is
    // the hard limit may be smaller when a limited domain
    int hard_limit = 0;
    if (model_type == enums::TRACK_MODEL) {
      int index = std::min(track_count, num_tracks-1);
      hard_limit = polyphony_hard_limits[index];
    }
    else {
      int index = std::min(infill_bar_count, num_infill_bars-1);
      hard_limit = polyphony_hard_limits[std::get<0>(selected_bars[index])];
    }

    // can't have more than n simultaneous notes
    if ((int)onsets.size() >= hard_limit) {
      data_structures::LOGGER(data_structures::to_str("HIT HARD LIMIT ( ",(int)onsets.size()," >= ", hard_limit, " ) >>>>>>>>>>>>>>>>>>>> "));
      rep->set_mask(midi::TOKEN_NOTE_ONSET, {-1}, mask, 0);
      rep->set_mask(midi::TOKEN_VELOCITY_LEVEL, {-1}, mask, 0);
      // will be ignored if token doesn't exist
    }

    //Check if microtiming is allowed
    int delta_domain_limit = rep->get_domain_size(midi::TOKEN_DELTA);

    if (!enc->config->use_microtiming) {
      for (int td=0; td<delta_domain_limit; td++) {
        data_structures::LOGGER(data_structures::VERBOSITY_LEVEL_TRACE, data_structures::to_str("NOT USE MICRO -> ","MASKING DELTA :: ", td));
        mask[rep->encode(midi::TOKEN_DELTA,td)] = 0;
      }
    } else {
      if (last_tt == midi::TOKEN_DELTA) {
        num_delta_tokens += 1;
        mask[rep->encode(midi::TOKEN_DELTA_DIRECTION,0)] = 0;
      } else {
        num_delta_tokens = 0;
      }
      //Check if max number microtiming tokens achieved
      if (num_delta_tokens > 0) {
        for (int td=0; td<delta_domain_limit; td++) {
          data_structures::LOGGER(data_structures::VERBOSITY_LEVEL_TRACE, data_structures::to_str("MAX MULTIPLE -> ","MASKING DELTA :: ", td));
          mask[rep->encode(midi::TOKEN_DELTA,td)] = 0;
        }
      }
      
      //Keep delta within the bar when forward
      if (delta_domain_limit) {
        int max_step = enc->config->step_to_delta(barlength - timestep, enc->config->resolution);
        if ((last_tt == midi::TOKEN_DELTA) && (rep->decode(last_token) == 0)){
          max_step = enc->config->step_to_delta(timestep, enc->config->resolution);
        }
        int max_td = std::max(std::min(max_step, delta_domain_limit), 0);
        for (int td=max_td; td<delta_domain_limit; td++) {
          data_structures::LOGGER(data_structures::VERBOSITY_LEVEL_TRACE, data_structures::to_str("MAX FORWARD/BACKWARD -> ","MASKING DELTA :: ", td));
          mask[rep->encode(midi::TOKEN_DELTA,td)] = 0;
        }
      }

      //Forward delta only at start of bar
      if (timestep == 0) {
        data_structures::LOGGER(data_structures::VERBOSITY_LEVEL_TRACE, data_structures::to_str("AT START -> ","MASKING DELTA DIRECTION"));
        mask[rep->encode(midi::TOKEN_DELTA_DIRECTION,0)] = 0;
      }

      //Backward delta only at end of bar
      if (timestep == barlength) {
        for (int td=1; td<delta_domain_limit; td++) {
          data_structures::LOGGER(data_structures::VERBOSITY_LEVEL_TRACE, data_structures::to_str("AT END -> ","MASKING DELTA :: ", 0));
          mask[rep->encode(midi::TOKEN_DELTA,td)] = 0;
        }
      }
    }
  
    // also restrict this for absolute time
    // but this should be limited based on time_signature / barlength
    int domain_limit = rep->get_domain_size(midi::TOKEN_TIME_ABSOLUTE_POS);
    if (domain_limit) {
      for (int td=0; td<=timestep; td++) {
        mask[rep->encode(midi::TOKEN_TIME_ABSOLUTE_POS,td)] = 0;
      }
      for (int td=barlength+1; td<domain_limit; td++) {
        mask[rep->encode(midi::TOKEN_TIME_ABSOLUTE_POS,td)] = 0;
      }
    }
    
    if (model_type == enums::TRACK_MODEL) {
      // limit number of bars
      if (bar_count != num_bars) {
        rep->set_mask(midi::TOKEN_TRACK_END, {-1}, mask, 0);
      }
      else {
        rep->set_mask(midi::TOKEN_BAR, {-1}, mask, 0);
      }
      // limit the track count
      if (track_count >= num_tracks) {
        std::fill(mask.begin(), mask.end(), 0);
        finished = true;
      }
      // only add attribute mask if not finished
      // otherwise it will crash with track_count out of range
      if (!finished) {
        for (int i=0; i<(int)mask.size(); i++) {
          int num_bars = attribute_bar_masks[track_count].size();
          int safe_bar_index = std::min(bar_count, num_bars - 1);
          mask[i] *= attribute_bar_masks[track_count][safe_bar_index][i];
        }
      }
    }
    else if ((model_type == enums::BAR_INFILL_MODEL)) {
      // limit the bar infill count
      if (infill_bar_count >= num_infill_bars) {
        std::fill(mask.begin(), mask.end(), 0);
        finished = true;
      }
    }

    // if mask is all zeros we have a problem as the model has
    // no 'valid' path forward
    if ((std::find(mask.begin(), mask.end(), 1) == mask.end()) && (!finished)) {
      throw std::runtime_error("FATAL ERROR : EVERY TOKEN IS MASKED");
    }

  }

  std::vector<int> get_mask(std::vector<int> &tokens) {
      data_structures::LOGGER(data_structures::VERBOSITY_LEVEL_TRACE, "get_mask" );
    std::vector<int> mask(enc->rep->max_token(), 0);
    for (int t=token_position; t<(int)tokens.size(); t++) {
      if (verbose) {
        data_structures::LOGGER(data_structures::to_str("UPDATING [", token_position, "] :: ", enc->rep->pretty(tokens[t])));
      }
      update( tokens[t] );
      history.push_back( tokens[t] );
      token_position++;
    }

    set_mask(tokens.back(), mask);
    return mask;
  }

  // map from pitch to when it expires
  std::map<int,std::vector<int>> note_expiry; // time -> list of pitches
  std::set<int> onsets;
  int last_token;
  int num_delta_tokens;
  midi::TRACK_TYPE current_track_type;

  int barlength;
  int timestep;
  int absolute_timestep;
  int bar_start_timestep;
  
  int bar_count;
  int track_count;
  int infill_bar_count;
  
  int num_bars;
  int num_tracks;
  int num_infill_bars;
  std::vector<std::vector<int>> attribute_masks;
  std::vector<std::vector<std::vector<int>>> attribute_bar_masks;

  std::vector<int> prompt;
  std::vector<int> inverse_order;
  std::string ckpt_path;

  int token_position;
  bool finished;
  enums::MODEL_TYPE model_type;
  std::vector<int> history;

  bool verbose;
  int polyphony_hard_limit;
  std::vector<int> polyphony_hard_limits;
  std::vector<float> track_temperatures;
  std::vector<std::pair<int,int>> selected_bars;

  std::unique_ptr<encoder::ENCODER> enc;
  std::shared_ptr<encoder::REPRESENTATION> rep;
  std::unique_ptr<REP_GRAPH> rg;
  std::unique_ptr<INSTRUMENT_CONDITIONAL_REP_GRAPH> instrument_rg;
  std::unique_ptr<DRUM_CONDITIONAL_REP_GRAPH> drum_rg;

};


}