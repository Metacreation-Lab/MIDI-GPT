#pragma once

#include "encoder_base.h"
#include "util.h"
#include "attribute_control.h"
#include "../data_structures/track_type.h"
#include "../../inference/enum/velocity.h"
#include "../../inference/enum/timesigs.h"
#include "../../inference/enum/pretrain_group.h"
#include "../midi_parsing/util_protobuf.h"
#include "../../inference/protobuf/validate.h"

// START OF NAMESPACE
namespace encoder {

template <typename T>
std::vector<T> operator+(std::vector<T> const &x, std::vector<T> const &y) {
  std::vector<T> vec;
  vec.reserve(x.size() + y.size());
  vec.insert(vec.end(), x.begin(), x.end());
  vec.insert(vec.end(), y.begin(), y.end());
  return vec;
}

class ExpressiveEncoder : public ENCODER {
public:
  ExpressiveEncoder() {
    config = std::make_shared<data_structures::EncoderConfig>();
    config->both_in_one = true;
    config->use_velocity_levels = true;
    config->use_microtiming = true;
    config->resolution = 12;
    config->delta_resolution = 1920;
    config->decode_resolution = config->delta_resolution;

    rep = std::make_shared<REPRESENTATION>(REPRESENTATION({
      {midi::TOKEN_PIECE_START, TOKEN_DOMAIN(2)},
      {midi::TOKEN_NUM_BARS, TOKEN_DOMAIN({4,8}, INT_VALUES_DOMAIN)},
      {midi::TOKEN_BAR, TOKEN_DOMAIN(1)},
      {midi::TOKEN_BAR_END, TOKEN_DOMAIN(1)},
      {midi::TOKEN_TIME_SIGNATURE, TOKEN_DOMAIN(
        enums::YELLOW_TS_MAP,TIMESIG_MAP_DOMAIN)},
      {midi::TOKEN_TRACK, TOKEN_DOMAIN({
        midi::STANDARD_TRACK,
        midi::STANDARD_DRUM_TRACK  
      },INT_VALUES_DOMAIN)},
      {midi::TOKEN_TRACK_END, TOKEN_DOMAIN(1)},
      {midi::TOKEN_INSTRUMENT, TOKEN_DOMAIN(enums::PRETRAIN_GROUPING,INT_MAP_DOMAIN)},
      {midi::TOKEN_NOTE_ONSET, TOKEN_DOMAIN(128)},
      {midi::TOKEN_NOTE_DURATION, TOKEN_DOMAIN(96)},
      {midi::TOKEN_TIME_ABSOLUTE_POS, TOKEN_DOMAIN(192)},
      {midi::TOKEN_FILL_IN_PLACEHOLDER, TOKEN_DOMAIN(1)},
      {midi::TOKEN_FILL_IN_START, TOKEN_DOMAIN(1)},
      {midi::TOKEN_FILL_IN_END, TOKEN_DOMAIN(1)},
      {midi::TOKEN_DELTA, TOKEN_DOMAIN(96)},
      {midi::TOKEN_DELTA_DIRECTION, TOKEN_DOMAIN(1)},
      {midi::TOKEN_VELOCITY_LEVEL, TOKEN_DOMAIN(128)},

      add_attribute_control_to_representation(midi::TOKEN_MIN_NOTE_DURATION),
      add_attribute_control_to_representation(midi::TOKEN_MAX_NOTE_DURATION),
      add_attribute_control_to_representation(midi::TOKEN_MIN_POLYPHONY),
      add_attribute_control_to_representation(midi::TOKEN_MAX_POLYPHONY),
      add_attribute_control_to_representation(midi::TOKEN_DENSITY_LEVEL),
    }));

  }
  ~ExpressiveEncoder() {}

  void preprocess_piece(midi::Piece *p) {
    util_protobuf::calculate_note_durations(p);
    util_protobuf::update_av_polyphony_and_note_duration(p);
    util_protobuf::update_note_density(p);
  }

  void set_scheme(int res, int delta_res, int delta_vocab_size, int abs_pos_vocab_size) {
    config->resolution = res;
    config->delta_resolution = delta_res;

    rep = std::make_shared<REPRESENTATION>(REPRESENTATION({
      {midi::TOKEN_PIECE_START, TOKEN_DOMAIN(2)},
      {midi::TOKEN_NUM_BARS, TOKEN_DOMAIN({4,8}, INT_VALUES_DOMAIN)},
      {midi::TOKEN_BAR, TOKEN_DOMAIN(1)},
      {midi::TOKEN_BAR_END, TOKEN_DOMAIN(1)},
      {midi::TOKEN_TIME_SIGNATURE, TOKEN_DOMAIN(
        enums::YELLOW_TS_MAP,TIMESIG_MAP_DOMAIN)},
      {midi::TOKEN_TRACK, TOKEN_DOMAIN({
        midi::STANDARD_TRACK,
        midi::STANDARD_DRUM_TRACK  
      },INT_VALUES_DOMAIN)},
      {midi::TOKEN_TRACK_END, TOKEN_DOMAIN(1)},
      {midi::TOKEN_INSTRUMENT, TOKEN_DOMAIN(enums::PRETRAIN_GROUPING,INT_MAP_DOMAIN)},
      {midi::TOKEN_NOTE_ONSET, TOKEN_DOMAIN(128)},
      {midi::TOKEN_NOTE_DURATION, TOKEN_DOMAIN(96)},
      {midi::TOKEN_TIME_ABSOLUTE_POS, TOKEN_DOMAIN(abs_pos_vocab_size)},
      {midi::TOKEN_FILL_IN_PLACEHOLDER, TOKEN_DOMAIN(1)},
      {midi::TOKEN_FILL_IN_START, TOKEN_DOMAIN(1)},
      {midi::TOKEN_FILL_IN_END, TOKEN_DOMAIN(1)},
      {midi::TOKEN_DELTA, TOKEN_DOMAIN(delta_vocab_size)},
      {midi::TOKEN_DELTA_DIRECTION, TOKEN_DOMAIN(1)},

      add_attribute_control_to_representation(midi::TOKEN_MIN_NOTE_DURATION),
      add_attribute_control_to_representation(midi::TOKEN_MAX_NOTE_DURATION),
      add_attribute_control_to_representation(midi::TOKEN_MIN_POLYPHONY),
      add_attribute_control_to_representation(midi::TOKEN_MAX_POLYPHONY),
      add_attribute_control_to_representation(midi::TOKEN_DENSITY_LEVEL),

      {midi::TOKEN_VELOCITY_LEVEL, TOKEN_DOMAIN(128)}
    }));
  }
};

}
// END OF NAMESPACE