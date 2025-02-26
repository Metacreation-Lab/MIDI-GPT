#pragma once

#include <assert.h>
#include <algorithm>

#include "callback_base.h"
#include "sample_internal.h"
#include "../../common/midi_parsing/util_protobuf.h"

#include <google/protobuf/util/message_differencer.h>
#include <torch/script.h>

#include "multi_step.h"

namespace sampling {

// Converts the status message into a track & bar matrix indicating which bars are selected
std::vector<std::vector<bool>> status_to_selection_mask(midi::Status *status) {
  data_structures::LOGGER(data_structures::VERBOSITY_LEVEL_TRACE, "status_to_selection_mask" );
  int ntracks = status->tracks_size();
  int nbars = status->tracks(0).selected_bars_size();
  std::vector<std::vector<bool>> x(ntracks, std::vector<bool>(nbars,false));
  int track_num = 0;
  for (const auto &track : status->tracks()) {
    int bar_num = 0;
    for (const auto &bar : track.selected_bars()) {
      x[track_num][bar_num] = bar;
      bar_num++;
    }
    track_num++;
  }
  return x;
}

// Returns a boolean vector indicating which tracks to sample
std::vector<bool> status_to_resample_mask(midi::Status *status) {
	data_structures::LOGGER(data_structures::VERBOSITY_LEVEL_TRACE, "status_to_resample_mask" );
  // get a boolean vector that indicates which tracks to resample
  std::vector<bool> resample_mask;
  for (const auto &track : status->tracks()) {
    resample_mask.push_back( track.autoregressive() );
  }
  return resample_mask;
}

// Returns a boolean vector indicating which tracks to ignore
std::vector<bool> status_to_ignore_mask(midi::Status *status) {
    data_structures::LOGGER(data_structures::VERBOSITY_LEVEL_TRACE, "status_to_ignore_mask" );
  std::vector<bool> ignore_mask;
  for (const auto &track : status->tracks()) {
    ignore_mask.push_back( track.ignore() );
  }
  return ignore_mask;
}


void status_rehighlight(midi::Status *status, const std::set<std::tuple<int,int>> &bar_list) {
  int num_tracks = status->tracks_size();
  for (int track_num=0; track_num<num_tracks; track_num++) {
    midi::StatusTrack *track = status->mutable_tracks(track_num);
    int num_bars = track->selected_bars_size();
    track->clear_selected_bars();
    for (int bar_num=0; bar_num<num_bars; bar_num++) {
      bool x = bar_list.find(std::make_tuple(track_num,bar_num)) != bar_list.end();
      track->add_selected_bars(x);
      if ((track->autoregressive()) && (!x)) {
        track->set_autoregressive( false );
      }
    }
  }
}

midi::Status status_subset(midi::Status *status, int start_bar, int end_bar, const std::vector<int> &track_indices) {
  midi::Status subset;
  subset.set_decode_final(status->decode_final());
  int track_count = 0;
  for (const auto &track_index : track_indices) {
    const midi::StatusTrack track = status->tracks(track_index);
    midi::StatusTrack *t = subset.add_tracks();
    t->CopyFrom(track);
    t->set_track_id(track_count);
    t->clear_selected_bars();
    t->clear_bars();
    for (int i=start_bar; i<end_bar; i++) {
      midi::StatusBar *b = t->add_bars();
      b->CopyFrom(track.bars(i));
      t->add_selected_bars( track.selected_bars(i) );
    }
    track_count++;
  }
  return subset;
}

// Retrieve a subset of the Piece
midi::Piece piece_subset(midi::Piece* piece, int start_bar, int end_bar, const std::vector<int>& track_indices) {
  midi::Piece subset;
  subset.set_resolution( piece->resolution() );
  subset.set_tempo( piece->tempo() );
  int track_count = 0;
  for (const auto &track_index : track_indices) {
    if (track_index >= piece->tracks_size()) {
      throw std::runtime_error("TRYING TO ACCESS TRACK OUT OF RANGE. PIECE IS LIKELY MALFORMED");
    }
    const midi::Track track = piece->tracks(track_index);
    midi::Track *t = subset.add_tracks();
    t->CopyFrom(track);
    t->clear_bars();
    for (int i=start_bar; i<end_bar; i++) {
      midi::Bar *b = t->add_bars();
      b->CopyFrom( track.bars(i) );
      b->clear_events();

      for (const auto &event : track.bars(i).events()) {
        b->add_events( subset.events_size() );
        midi::Event *e = subset.add_events();
        e->CopyFrom( piece->events(event) );
      }
    }
    track_count++;
  }
  return subset;
}

void add_timesigs_to_status(midi::Piece *piece, midi::Status *status) {
  data_structures::LOGGER(data_structures::VERBOSITY_LEVEL_TRACE, "add_timesigs_to_status" );
  int track_num = 0;
  for (const auto &track : piece->tracks()) {
    int bar_num = 0;
    midi::StatusTrack *st = status->mutable_tracks(track_num);
    for (const auto &bar : track.bars()) {
      midi::StatusBar *sb;
      if (st->bars_size() <= bar_num) {
        sb = st->add_bars();
      }
      else {
        sb = st->mutable_bars(bar_num);
      }
      sb->set_ts_numerator( bar.ts_numerator() );
      sb->set_ts_denominator( bar.ts_denominator() );
      bar_num++;
    }
    track_num++;
  }
}

// We compute features first and then only override if the controls are not "ANY"
void override_piece_features(midi::Piece *piece, midi::Status *status, const std::shared_ptr<encoder::REPRESENTATION> &rep) {
  data_structures::LOGGER(data_structures::VERBOSITY_LEVEL_TRACE, "override_piece_features" );
  compute_attribute_controls(rep, piece);

  // new override
  override_attribute_controls(rep, piece, status);

  // legacy override
  for (const auto &track : status->tracks()) {
    midi::TrackFeatures *f = util_protobuf::GetTrackFeatures(piece, track.track_id());
    if (track.density() > 0) {
      f->set_note_density_v2( track.density() - 1);
    }
    if (track.min_polyphony_q() > 0) {
      f->set_min_polyphony_q( track.min_polyphony_q() - 1 );
    }
    if (track.max_polyphony_q() > 0) {
      f->set_max_polyphony_q( track.max_polyphony_q() - 1 );
    }
    if (track.min_note_duration_q() > 0) {
      f->set_min_note_duration_q( track.min_note_duration_q() - 1 );
    }
    if (track.max_note_duration_q() > 0) {
      f->set_max_note_duration_q( track.max_note_duration_q() - 1 );
    }
  }
}

void piece_insert(midi::Piece *piece, midi::Piece *x, const std::vector<std::tuple<int,int,int,int>> &bar_mapping, bool verbose) {
    data_structures::LOGGER(data_structures::VERBOSITY_LEVEL_TRACE, "piece_insert" );

  for (const auto &ii : bar_mapping) {
    if (std::get<0>(ii) >= x->tracks_size()) {
      data_structures::LOGGER(data_structures::to_str("PIECE INSERT :: INVALID TRACK INDEX ", std::get<0>(ii), " FOR X"));
      throw std::runtime_error("PIECE INSERT :: INVALID TRACK INDEX FOR X");
    }
    if (std::get<2>(ii) >= piece->tracks_size()) {
      throw std::runtime_error("PIECE INSERT :: INVALID TRACK INDEX FOR PIECE");
    }
    const midi::Track src_track = x->tracks(std::get<0>(ii));
    const midi::Bar src = src_track.bars(std::get<1>(ii));
    midi::Track *dst_track = piece->mutable_tracks(std::get<2>(ii));
    midi::Bar *dst = dst_track->mutable_bars(std::get<3>(ii));

    if (verbose) {
      data_structures::LOGGER(data_structures::to_str("INSERTING (", std::get<0>(ii), ",", std::get<1>(ii), ") into (", std::get<2>(ii), ",", std::get<3>(ii), ")"));
    }

    // overwrite instrument and track type (for autoregressive)
    dst_track->set_track_type( src_track.track_type() );
    dst_track->set_instrument( src_track.instrument() );

    // overwrite bar from src
    dst->clear_events();
    for (const auto &event_index : src.events()) {
      dst->add_events( piece->events_size() );
      midi::Event *e = piece->add_events();
      e->CopyFrom( x->events(event_index) );
    }
  }
}

// This function resamples and recomputes the event times using the delta values
void resample_delta(midi::Piece *p, std::shared_ptr<data_structures::EncoderConfig> ec) {
  data_structures::LOGGER(data_structures::VERBOSITY_LEVEL_VERBOSE, "Resampling Piece with Delta values");
  int current_res = ec->resolution;
  int target_res = ec->decode_resolution;
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


std::vector<STEP> find_steps(const std::vector<std::vector<bool>> &sel, const std::vector<bool> &resample_mask, const std::vector<bool> &ignore_mask, midi::HyperParam *param) {
  if ((sel.size() != resample_mask.size()) || (sel.size() != ignore_mask.size())) {
    throw std::invalid_argument("find_steps :: selection, resample_mask and ignore_mask must be the same size");
  }
  std::vector<STEP> steps;
  cmatrix<bool> selection(sel);
  cmatrix<bool> generated = cmatrix<bool>(selection.N, selection.M, 0);
  cmatrix<bool> resample = vector_to_matrix(resample_mask, selection.M);
  cmatrix<bool> ignore = vector_to_matrix(ignore_mask, selection.M);
  find_steps_inner(steps, selection, resample, ignore, true, generated, param);
  find_steps_inner(steps, selection, resample, ignore, false, generated, param);
  return steps;
}

void sample_step(midi::Piece *piece, midi::Status *status, midi::HyperParam *param, const std::unique_ptr<ModelMeta> &model, const STEP *s, CallbackManager *callbacks) {
    data_structures::LOGGER(data_structures::VERBOSITY_LEVEL_TRACE, "sample_step" );
    
    // prepare the inputs for generation
    midi::Piece step_piece = piece_subset(piece, s->start, s->end, s->get_tracks());
    midi::Status step_status = status_subset(status, s->start, s->end, s->get_tracks());
    status_rehighlight(&step_status, s->get_bars_to_generate());  

    // do generation
    midi::Piece gen_piece = generate(&step_status, &step_piece, param, model, callbacks)[0];
    // NOTE : this inserts tracks that are just conditioned on as well
    // insert generation into global piece
    piece_insert(piece, &gen_piece, s->get_bar_mapping(), param->verbose());
    std::unique_ptr<encoder::ENCODER> enc = enums::getEncoderFromString(model->meta.encoder());
    if (!enc.get()) {
        throw std::invalid_argument("INVALID ENCODER");
    }
    if (enc->config->use_microtiming && status->decode_final()) {
      //resample_delta(piece, enc->config);
      enc->resample_delta(piece);
    }
    override_piece_features(piece, status, enc->rep);
}

// ==============================
// MAIN INFERENCE ENTRYPOINT
void sample(midi::Piece* piece, midi::Status* raw_status, midi::HyperParam* param, CallbackManager *callbacks) {
    data_structures::LOGGER(data_structures::VERBOSITY_LEVEL_TRACE, "sample" );

    //CheckIfDataExists
    if ((!piece) || (!raw_status) || (!param)) {
        throw std::invalid_argument("Piece, Status or HyperParam is malformed");
    }

    if ((callbacks) && (callbacks->is_cancelled())) {
      return;
    }

    // We create a new status with raw_status info, and then a pointer to access it indirectly.
    midi::Status status_object(*raw_status);
    midi::Status* status_pointer = &status_object;

    // try to load model
    std::unique_ptr<ModelMeta> model = load_model(param);

    // Check if encoder exists
    std::unique_ptr<encoder::ENCODER> enc = enums::getEncoderFromString(model->meta.encoder());
    if (!enc.get()) {
        throw std::invalid_argument("INVALID ENCODER");
    }
    piece->set_resolution(enc->config->resolution);
    param->set_internal_skip_preprocess(true);
    param->set_batch_size(1);

    util_protobuf::validate_inputs(piece, status_pointer, param);
    // before we start pad the piece if status references tracks
    // that do not exist yet
    util_protobuf::pad_piece_with_status(piece, status_pointer, param->model_dim());
    // add time-signatures from piece into the status
    add_timesigs_to_status(piece, status_pointer);
    // add features to piece when we are sampling auto-regressively
    // as these are perhaps not yet in the piece
    override_piece_features(piece, status_pointer, enc->rep);

    std::vector<std::vector<bool>> selection_mask = status_to_selection_mask(status_pointer);
    if (!any(selection_mask)) {
        return; // nothing to do
    }

    std::vector<bool> resample_mask = status_to_resample_mask(status_pointer);
    std::vector<bool> ignore_mask = status_to_ignore_mask(status_pointer);
    std::vector<STEP> steps = find_steps(selection_mask, resample_mask, ignore_mask, param);

    if (steps.size() == 0) {
        return; // nothing to be done
    }

    // find the total number of bars to be generated
    int bar_count = 0;
    for (const auto &step : steps) {
        bar_count += step.generated_bar_count();
    }

    // get order and reverse order of tracks
    int nt = status_pointer->tracks_size();
    std::vector<int> order(nt, 0);
    std::vector<int> reverse_order = arange(nt);
    for (int track_num = 0; track_num < nt; track_num++) {
        midi::StatusTrack* st = status_pointer->mutable_tracks(track_num);
        order[st->track_id()] = track_num;
        st->set_track_id(track_num); // now the mapping is the identity
    }
    std::sort(reverse_order.begin(), reverse_order.end(),
        [&order](size_t i, size_t j) {return order[i] < order[j]; });
    util_protobuf::reorder_tracks(piece, order);

    for (int i=0; i<steps.size(); i++) {
      if (i == steps.size() - 1) {
        status_pointer->set_decode_final(true);
      } else {
        status_pointer->set_decode_final(false);
      }
      STEP step = steps[i];
      data_structures::LOGGER(data_structures::VERBOSITY_LEVEL_TRACE, data_structures::to_str("Sampling step :: decoding final = ", status_pointer->decode_final()));
      sample_step(piece, status_pointer, param, model, &step, callbacks);
    }
    util_protobuf::reorder_tracks(piece, reverse_order);
    std::string json_string_res = util_protobuf::protobuf_to_string(piece);
}

std::vector<std::tuple<int,int,int>> get_notes_py(std::string &piece_json, int track_start, int track_end, int bar_start, int bar_end, bool onset_only_drums) {
  midi::Piece piece;
  util_protobuf::string_to_protobuf(piece_json, &piece);
  std::vector<midi::Note> notes = util_protobuf::getNotes(&piece, track_start, track_end, bar_start, bar_end, onset_only_drums);
  std::vector<std::tuple<int,int,int>> notes_py;
  for (const auto &note : notes) {
    notes_py.push_back(std::make_tuple(note.start(), note.end(), note.pitch()));
  }
  return notes_py;
}

void sort_notes(std::vector<midi::Note> &notes) {
  std::sort(notes.begin(), notes.end(), [](const midi::Note &a, const midi::Note &b) {
    if (a.start() == b.start()) {
      return a.pitch() < b.pitch();
    }
    return a.start() < b.start();
  });
}

// function that determines if two bars are equivalent
bool bars_are_equivalent(midi::Piece *pa, midi::Piece *pb, int track_num, int bar_num) {
  std::vector<midi::Note> notes_a = util_protobuf::getNotes(pa, track_num, track_num+1, bar_num, bar_num+1, true);
  std::vector<midi::Note> notes_b = util_protobuf::getNotes(pb, track_num, track_num+1, bar_num, bar_num+1, true);
  if (notes_a.size() != notes_b.size()) {
    return false;
  }
  sort_notes(notes_a);
  sort_notes(notes_b);
  for (int i=0; i<(int)notes_a.size(); i++) {
    if ((notes_a[i].start() != notes_b[i].start()) || (notes_a[i].pitch() != notes_b[i].pitch())) {
      return false;
    }
  }
  return true;
}

// function that determines if something has changed
// it returns a list of bars that are identical
std::vector<std::tuple<int,int>> find_identical_bars(midi::Piece *input, midi::Piece *output, midi::Status *status) {
  std::vector<std::tuple<int,int>> identical_bars;
  for (int track_num=0; track_num<status->tracks_size(); track_num++) {
    midi::StatusTrack track = status->tracks(track_num);
    for (int bar_num=0; bar_num<track.bars_size(); bar_num++) {
      if (track.selected_bars(bar_num)) {
        if (bars_are_equivalent(input, output, track_num, bar_num)) {
          identical_bars.push_back(std::make_tuple(track_num, bar_num));
        }
      }
    }
  }
  return identical_bars;
}

// wrapper function that ensures novelty and non-silence
int sample_multi_attempts(midi::Piece* piece, midi::Status* status, midi::HyperParam* param, CallbackManager *callbacks, int max_attempts) {
  int attempts = 0;
  midi::Piece input;
  input.CopyFrom(*piece);
  while (attempts < max_attempts) {
    std::cout << "ATTEMPT " << attempts << std::endl;
    midi::Piece current;
    current.CopyFrom(*piece);
    sample(&current, status, param, callbacks);
    std::vector<std::tuple<int,int>> identical_bars = find_identical_bars(&input, &current, status);
    attempts++;
    if (identical_bars.size() == 0) {
      piece->CopyFrom(current);
      return attempts;
    }
    if (callbacks) {
      param->set_temperature( callbacks->update_temperature(param->temperature()) );
    }
  }
  return attempts;
}

std::tuple<std::string,int> sample_multi_step_py(std::string &piece_json, std::string &status_json, std::string &param_json, int max_attempts, sampling::CallbackManager *callbacks) {
  midi::Piece piece;
  midi::Status status;
  midi::HyperParam hyperParam;

  data_structures::LOGGER(data_structures::VERBOSITY_LEVEL_TRACE, "to_proto");

  util_protobuf::string_to_protobuf(piece_json, &piece);
  util_protobuf::string_to_protobuf(status_json, &status);
  util_protobuf::string_to_protobuf(param_json, &hyperParam);
  data_structures::LOGGER(data_structures::VERBOSITY_LEVEL_TRACE, "validating");

  util_protobuf::validate_protobuf_fields(&piece, piece_json);
  data_structures::LOGGER(data_structures::VERBOSITY_LEVEL_TRACE, "piece");
  util_protobuf::validate_protobuf_fields(&status, status_json);
  data_structures::LOGGER(data_structures::VERBOSITY_LEVEL_TRACE, "status");
  util_protobuf::validate_protobuf_fields(&hyperParam, param_json);
  data_structures::LOGGER(data_structures::VERBOSITY_LEVEL_TRACE, "param");

  data_structures::LOGGER(data_structures::VERBOSITY_LEVEL_VERBOSE, util_protobuf::protobuf_to_string(&status));
  data_structures::LOGGER(data_structures::VERBOSITY_LEVEL_VERBOSE, util_protobuf::protobuf_to_string(&hyperParam));

  int attempts = sample_multi_attempts(&piece, &status, &hyperParam, callbacks, max_attempts);
  return std::make_tuple(util_protobuf::protobuf_to_string(&piece), attempts);
}

}
