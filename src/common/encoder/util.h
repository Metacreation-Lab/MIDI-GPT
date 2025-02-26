#pragma once

#include <google/protobuf/util/json_util.h>

#include <vector>
#include <map>
#include <tuple>
#include <array>
#include <utility>
#include <random>

#include "midi.pb.h"
#include "../midi_parsing/util_protobuf.h"
#include "attribute_control.h"
#include "../../inference/enum/constants.h"
#include "../../common/data_structures/encoder_config.h"
#include "../../common/data_structures/token_sequence.h"
#include "../../inference/enum/pretrain_group.h"

// START OF NAMESPACE
namespace encoder {


// below is a simplified refactor of the encoding process
// broken into clear functions to
// - encode notes within a bar
// - encode a bar
// - encode a track
// - encode a piece

void decode_track(std::vector<int> &tokens, midi::Piece *p, const std::shared_ptr<encoder::REPRESENTATION> &rep, const std::shared_ptr<data_structures::EncoderConfig> &ec) {
  p->set_resolution(ec->resolution);

  std::map<int,int> inst_to_track;
  midi::Event *e = NULL;
  midi::Track *t = NULL;
  midi::Bar *b = NULL;
  int current_time, current_note_time, current_instrument, delta_direction, delta_total;
  int beat_length = 0;
  int track_count = 0;
  int bar_count = 0;
  int last_token = -1;
  int last_abs_token = -1;
  int current_velocity = 100;

  std::set<int> offset_remain;

  for (const auto &token : tokens) {
    if (rep->is_token_type(token, midi::TOKEN_TRACK)) {
      current_time = 0; // restart the time
      current_note_time = 0;
      current_instrument = 0; // reset instrument
      delta_direction = 1;
      delta_total = 0;
      offset_remain.clear();
      if (track_count >= p->tracks_size()) {
        t = p->add_tracks();
      }
      else {
        t = p->mutable_tracks(track_count);
      }
      t->set_track_type( (midi::TRACK_TYPE)rep->decode(token) );
      util_protobuf::GetTrackFeatures(p, track_count);
    }
    else if (rep->is_token_type(token, midi::TOKEN_TRACK_END)) {
      track_count++;
      t = NULL;
    }
    else if (rep->is_token_type(token, midi::TOKEN_BAR)) {
      // when we start new bar we need to decrement time of remaining offsets
      for (const auto &index : offset_remain) {
        midi::Event *e = p->mutable_events(index);
        e->set_time( (int)(((e->time() - beat_length * ec->resolution)*(p->resolution())/ec->resolution)));
      }
      current_time = 0; // restart the time
      current_note_time = 0;
      delta_direction = 1;
      delta_total = 0;
      beat_length = 4; // default value optionally overidden with TIME_SIGNATURE
      if (t) {
        b = t->add_bars();
      }
      bar_count++;
    }
    else if (rep->is_token_type(token, midi::TOKEN_TIME_SIGNATURE)) {
      std::tuple<int,int> ts = rep->decode_timesig(token);
      beat_length = 4 * std::get<0>(ts) / std::get<1>(ts);
      b->set_ts_numerator( std::get<0>(ts) );
      b->set_ts_denominator( std::get<1>(ts) );
    }
    else if (rep->is_token_type(token, midi::TOKEN_BAR_END)) {
      if (b) {
        b->set_internal_beat_length(beat_length);
      }
      current_time = beat_length * p->resolution();
      current_note_time = current_time;
    }
    else if (rep->is_token_type(token, midi::TOKEN_TIME_ABSOLUTE_POS)) {
      current_time = rep->decode(token); // simply update instead of increment
      current_note_time = current_time;
      delta_direction = 1;
      delta_total = 0;
    }
    else if (rep->is_token_type(token, midi::TOKEN_DELTA_DIRECTION)) {
      delta_direction = -1;
      delta_total = 0;
    }
    else if (rep->is_token_type(token, midi::TOKEN_DELTA)) {
      last_abs_token = last_token;
      int delta_val = rep->decode(token);
      delta_total += delta_direction * delta_val;
      
    }
    else if (rep->is_token_type(token, midi::TOKEN_INSTRUMENT)) {
      if (t) {
        current_instrument = rep->decode(token);
        t->set_instrument( current_instrument );
      }
    }
    else if (rep->is_token_type(token, midi::TOKEN_VELOCITY_LEVEL)) {
      current_velocity = rep->decode(token);
    }
    else if (rep->is_token_type(token, midi::TOKEN_NOTE_ONSET)) {
      if (b && t) {
        
        if (data_structures::is_drum_track(t->track_type())) {
          
          int current_note_index = p->events_size();
          current_note_time = current_time;
          e = p->add_events();
          e->set_pitch( rep->decode(token) );
          e->set_velocity( current_velocity );
          e->set_time( current_note_time );

          e->set_delta( delta_total );
          delta_total = 0;
          delta_direction = 1;
          b->add_events( current_note_index );
          b->set_internal_has_notes( true );

          current_note_index = p->events_size();
          e = p->add_events();
          e->set_pitch( rep->decode(token) );
          e->set_velocity( 0 );
          e->set_time( current_note_time + 1 );
          b->add_events( current_note_index );
          b->set_internal_has_notes( true );

        }
      }
    }
    else if (rep->is_token_type(token, midi::TOKEN_NOTE_DURATION)) {
      if (b && t && (last_token >= 0) && (rep->is_token_type(last_token, midi::TOKEN_NOTE_ONSET))) {

        // add onset
        int current_note_index = p->events_size();
        current_note_time = current_time;
        e = p->add_events();
        e->set_pitch( rep->decode(last_token) );
        e->set_velocity( current_velocity );
        e->set_time( current_note_time );
        e->set_delta( delta_total );
        delta_total = 0;
        delta_direction = 1;
        b->add_events( current_note_index );

        // add offset
        current_note_index = p->events_size();
        e = p->add_events();
        e->set_pitch( rep->decode(last_token) );
        e->set_velocity( 0 );
        e->set_time( current_note_time + rep->decode(token) + 1 );
        e->set_delta( 0 );

        if (e->time() <= beat_length * p->resolution()) {
          b->add_events( current_note_index );
        }
        else {
          // we need to add this to a later bar
          offset_remain.insert( current_note_index );
        }

        b->set_internal_has_notes( true );
      }
    }
    else if (rep->is_token_type(token, midi::TOKEN_GENRE)) {
      midi::TrackFeatures *f;
      if (!t->internal_features_size()) {
        f = t->add_internal_features(); 
      }
      else {
        f = t->mutable_internal_features(0);
      }
      f->set_genre_str( rep->decode_string(token) );
    }

    // insert offsets from note_duration tokens when possible
    std::vector<int> to_remove;
    for (const auto &index : offset_remain) {
      if ((int)p->events(index).time()  <= current_time) {
        b->add_events( index );
        to_remove.push_back( index );
      }
    }
    for (const auto &index : to_remove) {
      offset_remain.erase(index);
    }

    last_token = token;
  }
  p->add_internal_valid_segments(0);
  p->add_internal_valid_tracks((1<<p->tracks_size())-1);
}

}
// END OF NAMESPACE
