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
    util_protobuf::preprocess_tracks(p);
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

class ElVelocityDurationPolyphonyYellowEncoder : public ENCODER {
public:
  ElVelocityDurationPolyphonyYellowEncoder() {
    config = std::make_shared<data_structures::EncoderConfig>();
    config->both_in_one = true;
    config->force_instrument = true;
    config->mark_note_duration_quantile = true;
    config->mark_polyphony_quantile = true;
    config->use_note_duration_encoding = true;
    config->use_absolute_time_encoding = true;
    config->mark_time_sigs = true;
    config->mark_drum_density = true;
    config->use_drum_offsets = false;
    config->use_velocity_levels = true;
    config->min_tracks = 1;
    config->resolution = 12;

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

      add_attribute_control_to_representation(midi::TOKEN_MIN_NOTE_DURATION),
      add_attribute_control_to_representation(midi::TOKEN_MAX_NOTE_DURATION),
      add_attribute_control_to_representation(midi::TOKEN_MIN_POLYPHONY),
      add_attribute_control_to_representation(midi::TOKEN_MAX_POLYPHONY),
      add_attribute_control_to_representation(midi::TOKEN_DENSITY_LEVEL),

      {midi::TOKEN_VELOCITY_LEVEL, TOKEN_DOMAIN(enums::DEFAULT_VELOCITY_MAP,INT_MAP_DOMAIN)}
    }));

  }
  ~ElVelocityDurationPolyphonyYellowEncoder() {}

  void preprocess_piece(midi::Piece *p) {
    util_protobuf::calculate_note_durations(p);
    util_protobuf::update_av_polyphony_and_note_duration(p);
    util_protobuf::update_note_density(p);
  }
};

// ================================================
// Shared representations and base classes
// ================================================

std::vector<std::pair<midi::TOKEN_TYPE,TOKEN_DOMAIN>> basic_rep = {
  {midi::TOKEN_PIECE_START, TOKEN_DOMAIN(2)},
  {midi::TOKEN_NUM_BARS, TOKEN_DOMAIN({4,8}, INT_VALUES_DOMAIN)},
  {midi::TOKEN_BAR, TOKEN_DOMAIN(1)},
  {midi::TOKEN_BAR_END, TOKEN_DOMAIN(1)},
  {midi::TOKEN_TIME_SIGNATURE, TOKEN_DOMAIN(enums::YELLOW_TS_MAP,TIMESIG_MAP_DOMAIN)},
  {midi::TOKEN_TRACK, TOKEN_DOMAIN({midi::STANDARD_TRACK,midi::STANDARD_DRUM_TRACK},INT_VALUES_DOMAIN)},
  {midi::TOKEN_TRACK_END, TOKEN_DOMAIN(1)},
  {midi::TOKEN_INSTRUMENT, TOKEN_DOMAIN(enums::PRETRAIN_GROUPING,INT_MAP_DOMAIN)},
  {midi::TOKEN_NOTE_ONSET, TOKEN_DOMAIN(128)},
  {midi::TOKEN_NOTE_DURATION, TOKEN_DOMAIN(96)},
  {midi::TOKEN_TIME_ABSOLUTE_POS, TOKEN_DOMAIN(192)},
  {midi::TOKEN_FILL_IN_PLACEHOLDER, TOKEN_DOMAIN(1)},
  {midi::TOKEN_FILL_IN_START, TOKEN_DOMAIN(1)},
  {midi::TOKEN_FILL_IN_END, TOKEN_DOMAIN(1)}
};

std::shared_ptr<data_structures::EncoderConfig> get_default_encoder_config() {
  auto e = std::make_shared<data_structures::EncoderConfig>();
  e->both_in_one = true;
  e->force_instrument = true;
  e->use_note_duration_encoding = true;
  e->use_absolute_time_encoding = true;
  e->mark_time_sigs = true;

  e->mark_note_duration_quantile = false;
  e->mark_polyphony_quantile = false;
  e->mark_drum_density = false;
  e->use_drum_offsets = false;
  e->use_velocity_levels = false;
  e->min_tracks = 1;
  e->resolution = 12;
  return e;
}

void build_rep_spec_v2(ENCODER *e, std::vector<std::pair<midi::TOKEN_TYPE,TOKEN_DOMAIN>> &spec, std::vector<midi::ATTRIBUTE_CONTROL_TYPE> controls) {
  std::vector<std::pair<midi::TOKEN_TYPE,TOKEN_DOMAIN>> output;
  std::copy(spec.begin(), spec.end(), std::back_inserter(output));
  for (auto c : controls) {
    for (auto t : add_attribute_control_to_representation_v2(c)) {
      output.push_back(t);
    }
    e->attribute_control_types.push_back(c);
  }
  e->rep = std::make_shared<REPRESENTATION>(REPRESENTATION(output));
}

class ENCODER_EXP_BASE : public ENCODER {
public:
  void preprocess_piece(midi::Piece *p) {
    util_protobuf::calculate_note_durations(p);
    compute_attribute_controls(this->rep, p);
  }
};

// GhostEncoder: based on ElVelocityDurationPolyphonyYellowEncoder with
// TOKEN_MASK_BAR added — enables lookahead generation and mask augmentation.
// "Ghost bars" represent unknown future content in conditioning tracks.
class GhostEncoder : public ENCODER {
public:
  GhostEncoder() {
    config = std::make_shared<data_structures::EncoderConfig>();
    config->both_in_one = true;
    config->force_instrument = true;
    config->mark_note_duration_quantile = true;
    config->mark_polyphony_quantile = true;
    config->use_note_duration_encoding = true;
    config->use_absolute_time_encoding = true;
    config->mark_time_sigs = true;
    config->mark_drum_density = true;
    config->use_drum_offsets = false;
    config->use_velocity_levels = true;
    config->min_tracks = 1;
    config->resolution = 12;

    rep = std::make_shared<REPRESENTATION>(REPRESENTATION({
      {midi::TOKEN_PIECE_START, TOKEN_DOMAIN(2)},
      {midi::TOKEN_NUM_BARS, TOKEN_DOMAIN({4,8,12,16}, INT_VALUES_DOMAIN)},
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
      {midi::TOKEN_MASK_BAR, TOKEN_DOMAIN(1)},

      add_attribute_control_to_representation(midi::TOKEN_MIN_NOTE_DURATION),
      add_attribute_control_to_representation(midi::TOKEN_MAX_NOTE_DURATION),
      add_attribute_control_to_representation(midi::TOKEN_MIN_POLYPHONY),
      add_attribute_control_to_representation(midi::TOKEN_MAX_POLYPHONY),
      add_attribute_control_to_representation(midi::TOKEN_DENSITY_LEVEL),

      {midi::TOKEN_VELOCITY_LEVEL, TOKEN_DOMAIN(enums::DEFAULT_VELOCITY_MAP,INT_MAP_DOMAIN)}
    }));
  }
  ~GhostEncoder() {}

  void preprocess_piece(midi::Piece *p) {
    util_protobuf::calculate_note_durations(p);
    util_protobuf::update_av_polyphony_and_note_duration(p);
    util_protobuf::update_note_density(p);
  }
};

// SpecterEncoder: Yellow encoder augmented with bar-level tonal tension for
// instrument tracks and bar-level rhythm tension for drum tracks.
//
// The tension algorithm is a faithful C++ port of the spiral-array tonal
// tension from tension_calculation.py (Chew 2002 / Guo / Farbood), so that
// C++ and Python produce numerically identical bin values for the same MIDI.
//
// Configurable weights (matching the Python repo defaults):
//   tension_config.onset_weight  = featureWeights_drums[0] = 2
//   tension_config.vel_weight    = featureWeights_drums[2] = 3
//   tension_config.n_bins        = domain size of the tension token = 10
class SpecterEncoder : public ENCODER {
public:
  // TensionConfig allows the caller to override the default weights
  // (matching the Python repo's featureWeights_drums) without changing
  // the mathematical algorithm.
  struct TensionConfig {
    double onset_weight = 2.0;   // drum: onset-frequency weight (featureWeights_drums[0])
    double vel_weight   = 3.0;   // drum: loudness weight          (featureWeights_drums[2])
    int    n_bins       = 10;    // bin count [0 .. n_bins-1] for both tension tokens
    bool   use_tension  = true;
  } tension_config;

  SpecterEncoder() {
    config = std::make_shared<data_structures::EncoderConfig>();
    config->both_in_one = true;
    config->force_instrument = true;
    config->mark_note_duration_quantile = true;
    config->mark_polyphony_quantile = true;
    config->use_note_duration_encoding = true;
    config->use_absolute_time_encoding = true;
    config->mark_time_sigs = true;
    config->mark_drum_density = true;
    config->use_drum_offsets = false;
    config->use_velocity_levels = true;
    config->min_tracks = 1;
    config->resolution = 12;

    rep = std::make_shared<REPRESENTATION>(REPRESENTATION({
      {midi::TOKEN_PIECE_START, TOKEN_DOMAIN(2)},
      {midi::TOKEN_NUM_BARS, TOKEN_DOMAIN({4,8}, INT_VALUES_DOMAIN)},   // Yellow range
      {midi::TOKEN_BAR, TOKEN_DOMAIN(1)},
      {midi::TOKEN_BAR_END, TOKEN_DOMAIN(1)},
      {midi::TOKEN_TIME_SIGNATURE, TOKEN_DOMAIN(enums::YELLOW_TS_MAP, TIMESIG_MAP_DOMAIN)},
      {midi::TOKEN_TRACK, TOKEN_DOMAIN({midi::STANDARD_TRACK, midi::STANDARD_DRUM_TRACK},
                                        INT_VALUES_DOMAIN)},
      {midi::TOKEN_TRACK_END, TOKEN_DOMAIN(1)},
      {midi::TOKEN_INSTRUMENT, TOKEN_DOMAIN(enums::PRETRAIN_GROUPING, INT_MAP_DOMAIN)},
      {midi::TOKEN_NOTE_ONSET, TOKEN_DOMAIN(128)},
      {midi::TOKEN_NOTE_DURATION, TOKEN_DOMAIN(96)},
      {midi::TOKEN_TIME_ABSOLUTE_POS, TOKEN_DOMAIN(192)},
      {midi::TOKEN_FILL_IN_PLACEHOLDER, TOKEN_DOMAIN(1)},
      {midi::TOKEN_FILL_IN_START, TOKEN_DOMAIN(1)},
      {midi::TOKEN_FILL_IN_END, TOKEN_DOMAIN(1)},

      add_attribute_control_to_representation(midi::TOKEN_MIN_NOTE_DURATION),
      add_attribute_control_to_representation(midi::TOKEN_MAX_NOTE_DURATION),
      add_attribute_control_to_representation(midi::TOKEN_MIN_POLYPHONY),
      add_attribute_control_to_representation(midi::TOKEN_MAX_POLYPHONY),
      add_attribute_control_to_representation(midi::TOKEN_DENSITY_LEVEL),

      // Bar-level tension — registered through attribute control system
      add_attribute_control_to_representation(midi::TOKEN_BAR_LEVEL_TENSION),
      add_attribute_control_to_representation(midi::TOKEN_BAR_LEVEL_TENSION_DRUM),

      {midi::TOKEN_VELOCITY_LEVEL, TOKEN_DOMAIN(enums::DEFAULT_VELOCITY_MAP, INT_MAP_DOMAIN)}
    }));
  }
  ~SpecterEncoder() {}

  void set_use_tension(bool use) {
    tension_config.use_tension = use;
  }

  void preprocess_piece(midi::Piece *p) {
    util_protobuf::calculate_note_durations(p);
    util_protobuf::update_av_polyphony_and_note_duration(p);
    util_protobuf::update_note_density(p);

    // Pre-compute tension with ported Farbood model and store in BarFeatures.
    // The attribute controls (BarLevelInstrumentTension, BarLevelDrumTension)
    // check has_tension() / has_tension_drum() and skip if already set here.
    if (tension_config.use_tension) {
      farbood::precompute_instrument_tension(p, tension_config.n_bins);
      farbood::precompute_drum_tension(p, tension_config.n_bins);
    }
    compute_attribute_controls(this->rep, p);
  }
};

// OracleEncoder: based on ElVelocityDurationPolyphonyYellowEncoder with
// TOKEN_MASK_BAR added and with bar-level tonal tension for
// instrument tracks and bar-level rhythm tension for drum tracks.
class OracleEncoder : public ENCODER {
public:

  struct TensionConfig {
    double onset_weight = 2.0;   // drum: onset-frequency weight (featureWeights_drums[0])
    double vel_weight   = 3.0;   // drum: loudness weight          (featureWeights_drums[2])
    int    n_bins       = 10;    // bin count [0 .. n_bins-1] for both tension tokens
    bool   use_tension  = true;
  } tension_config;

  OracleEncoder() {
    config = std::make_shared<data_structures::EncoderConfig>();
    config->both_in_one = true;
    config->force_instrument = true;
    config->mark_note_duration_quantile = true;
    config->mark_polyphony_quantile = true;
    config->use_note_duration_encoding = true;
    config->use_absolute_time_encoding = true;
    config->mark_time_sigs = true;
    config->mark_drum_density = true;
    config->use_drum_offsets = false;
    config->use_velocity_levels = true;
    config->min_tracks = 1;
    config->resolution = 12;

    rep = std::make_shared<REPRESENTATION>(REPRESENTATION({
      {midi::TOKEN_PIECE_START, TOKEN_DOMAIN(2)},
      {midi::TOKEN_NUM_BARS, TOKEN_DOMAIN({4,8,12,16}, INT_VALUES_DOMAIN)},
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
      {midi::TOKEN_MASK_BAR, TOKEN_DOMAIN(1)},

      add_attribute_control_to_representation(midi::TOKEN_MIN_NOTE_DURATION),
      add_attribute_control_to_representation(midi::TOKEN_MAX_NOTE_DURATION),
      add_attribute_control_to_representation(midi::TOKEN_MIN_POLYPHONY),
      add_attribute_control_to_representation(midi::TOKEN_MAX_POLYPHONY),
      add_attribute_control_to_representation(midi::TOKEN_DENSITY_LEVEL),

      // Bar-level tension — registered through attribute control system
      add_attribute_control_to_representation(midi::TOKEN_BAR_LEVEL_TENSION),
      add_attribute_control_to_representation(midi::TOKEN_BAR_LEVEL_TENSION_DRUM),

      {midi::TOKEN_VELOCITY_LEVEL, TOKEN_DOMAIN(enums::DEFAULT_VELOCITY_MAP,INT_MAP_DOMAIN)}
    }));
  }
  ~OracleEncoder() {}

  void set_use_tension(bool use) {
    tension_config.use_tension = use;
  }

  void preprocess_piece(midi::Piece *p) {
    util_protobuf::calculate_note_durations(p);
    util_protobuf::update_av_polyphony_and_note_duration(p);
    util_protobuf::update_note_density(p);

    // Pre-compute tension with ported Farbood model and store in BarFeatures.
    // The attribute controls (BarLevelInstrumentTension, BarLevelDrumTension)
    // check has_tension() / has_tension_drum() and skip if already set here.
    if (tension_config.use_tension) {
      farbood::precompute_instrument_tension(p, tension_config.n_bins);
      farbood::precompute_drum_tension(p, tension_config.n_bins);
    }
    compute_attribute_controls(this->rep, p);
  }
};

class SteinbergWPCSEncoder : public ENCODER_EXP_BASE {
public:
  SteinbergWPCSEncoder() {
    config = get_default_encoder_config();
    config->resolution = 24;
    std::vector<std::pair<midi::TOKEN_TYPE,TOKEN_DOMAIN>> spec;
    std::copy(basic_rep.begin(), basic_rep.end(), std::back_inserter(spec));
    spec.push_back({midi::TOKEN_VELOCITY_LEVEL, TOKEN_DOMAIN(enums::DEFAULT_VELOCITY_MAP,INT_MAP_DOMAIN)});
    build_rep_spec_v2(this, basic_rep, {
      midi::ATTRIBUTE_CONTROL_TRACK_LEVEL_ONSET_DENSITY,
      midi::ATTRIBUTE_CONTROL_TRACK_LEVEL_ONSET_POLYPHONY,
      midi::ATTRIBUTE_CONTROL_TRACK_LEVEL_NOTE_DURATION,
      midi::ATTRIBUTE_CONTROL_REPETITION,
      midi::ATTRIBUTE_CONTROL_GENRE,
      midi::ATTRIBUTE_CONTROL_BAR_LEVEL_PITCH_CLASS_SET
    });
  }
  ~SteinbergWPCSEncoder() {}
};

}
// END OF NAMESPACE