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
    
    // Find the first bar to be highlighted in this track
    int first_selected = -1;
    for (int bar_num=0; bar_num<num_bars; bar_num++) {
      if (bar_list.count(std::make_tuple(track_num, bar_num))) {
        first_selected = bar_num;
        break;
      }
    }

    track->clear_selected_bars();
    for (int bar_num=0; bar_num<num_bars; bar_num++) {
      bool x = bar_list.find(std::make_tuple(track_num,bar_num)) != bar_list.end();
      // If suffix-autoregressive is enabled, all bars from the first selected onwards must be TRUE
      if (track->suffix_autoregressive() && first_selected != -1 && bar_num >= first_selected) {
        x = true;
      }
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
      if (i < track.bars_size()) {
          b->CopyFrom(track.bars(i));
          t->add_selected_bars( track.selected_bars(i) );
      } else {
          t->add_selected_bars(false);
      }
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
      if (i < track.bars_size()) {
          b->CopyFrom( track.bars(i) );
          b->clear_events();

          for (const auto &event : track.bars(i).events()) {
            b->add_events( subset.events_size() );
            midi::Event *e = subset.add_events();
            e->CopyFrom( piece->events(event) );
          }
      }
      // If i >= track.bars_size(), we still added an empty bar above (t->add_bars())
      // to maintain index consistency with the generation window.
    }
    track_count++;
  }
  return subset;
}

// Copy StatusBar.future flags from the status into Bar.future in the piece.
// Called during preprocessing so the encoder sees the correct masking without
// requiring callers to mutate the piece JSON.  Only overwrites Bar.future when
// the corresponding StatusBar has the future field explicitly set (has_future()).
// Bars whose StatusBar does not set future are left unchanged, preserving any
// Bar.future values already present (e.g. from training augmentation).
void apply_future_flags_from_status(midi::Piece *piece, midi::Status *status) {
  for (int ti = 0; ti < status->tracks_size(); ti++) {
    const midi::StatusTrack &st = status->tracks(ti);
    int track_id = st.track_id();
    if (track_id < 0 || track_id >= piece->tracks_size()) continue;
    midi::Track *track = piece->mutable_tracks(track_id);
    int num_bars = std::min(st.bars_size(), track->bars_size());
    for (int bi = 0; bi < num_bars; bi++) {
      const midi::StatusBar &sb = st.bars(bi);
      if (sb.has_future()) {
        track->mutable_bars(bi)->set_future(sb.future());
      }
    }
  }
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
    if (std::get<1>(ii) >= src_track.bars_size()) {
      data_structures::LOGGER(data_structures::to_str("PIECE INSERT :: INVALID BAR INDEX ", std::get<1>(ii), " FOR SRC TRACK (size: ", src_track.bars_size(), ")"));
      throw std::runtime_error("PIECE INSERT :: INVALID BAR INDEX FOR X");
    }
    const midi::Bar src = src_track.bars(std::get<1>(ii));
    midi::Track *dst_track = piece->mutable_tracks(std::get<2>(ii));
    if (std::get<3>(ii) >= dst_track->bars_size()) {
      data_structures::LOGGER(data_structures::to_str("PIECE INSERT :: INVALID BAR INDEX ", std::get<3>(ii), " FOR DST TRACK (size: ", dst_track->bars_size(), ")"));
      throw std::runtime_error("PIECE INSERT :: INVALID BAR INDEX FOR DST");
    }
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

void sample_step(midi::Piece *piece, midi::Status *status, midi::HyperParam *param, const std::unique_ptr<ModelMeta> &model, const STEP *s, CallbackManager *callbacks, SamplingTimings *timings = nullptr) {
    data_structures::LOGGER(data_structures::VERBOSITY_LEVEL_TRACE, "sample_step" );
    if (timings) timings->steps++;

    // prepare the inputs for generation
    auto _t_slice = _Clock::now();
    midi::Piece step_piece = piece_subset(piece, s->start, s->end, s->get_tracks());
    midi::Status step_status = status_subset(status, s->start, s->end, s->get_tracks());
    status_rehighlight(&step_status, s->get_bars_to_generate());
    if (timings) timings->slice_ms += _ms(_t_slice);

    // do generation
    midi::Piece gen_piece = generate(&step_status, &step_piece, param, model, callbacks, timings)[0];

    // insert generation into global piece + postprocess
    auto _t_post = _Clock::now();
    piece_insert(piece, &gen_piece, s->get_bar_mapping(), param->verbose());
    std::unique_ptr<encoder::ENCODER> enc = enums::getEncoderFromString(model->meta.encoder());
    if (!enc.get()) {
        throw std::invalid_argument("INVALID ENCODER");
    }
    if (enc->config->use_microtiming && status->decode_final()) {
      enc->resample_delta(piece);
    }
    override_piece_features(piece, status, enc->rep);
    if (timings) timings->postprocess_ms += _ms(_t_post);
}

// ==============================
// MAIN INFERENCE ENTRYPOINT
void sample(midi::Piece* piece, midi::Status* raw_status, midi::HyperParam* param, CallbackManager *callbacks, SamplingTimings *timings = nullptr) {
    data_structures::LOGGER(data_structures::VERBOSITY_LEVEL_TRACE, "sample" );

    //CheckIfDataExists
    if ((!piece) || (!raw_status) || (!param)) {
        throw std::invalid_argument("Piece, Status or HyperParam is malformed");
    }

    if ((callbacks) && (callbacks->is_cancelled())) {
      return;
    }

    auto _t_total = _Clock::now();

    // We create a new status with raw_status info, and then a pointer to access it indirectly.
    midi::Status status_object(*raw_status);
    midi::Status* status_pointer = &status_object;

    // try to load model
    auto _t_load = _Clock::now();
    std::unique_ptr<ModelMeta> model = load_model(param);
    if (timings) timings->model_load_ms += _ms(_t_load);

    // Check if encoder exists
    std::unique_ptr<encoder::ENCODER> enc = enums::getEncoderFromString(model->meta.encoder());
    if (!enc.get()) {
        throw std::invalid_argument("INVALID ENCODER");
    }
    piece->set_resolution(enc->config->resolution);
    param->set_internal_skip_preprocess(true);
    param->set_batch_size(1);

    auto _t_pre = _Clock::now();
    util_protobuf::validate_inputs(piece, status_pointer, param);
    util_protobuf::pad_piece_with_status(piece, status_pointer, param->model_dim());
    add_timesigs_to_status(piece, status_pointer);
    apply_future_flags_from_status(piece, status_pointer);
    override_piece_features(piece, status_pointer, enc->rep);
    if (timings) timings->preprocess_ms += _ms(_t_pre);

    std::vector<std::vector<bool>> selection_mask = status_to_selection_mask(status_pointer);
    if (!any(selection_mask)) {
        if (timings) timings->total_gen_ms += _ms(_t_total);
        return;
    }

    std::vector<bool> resample_mask = status_to_resample_mask(status_pointer);
    std::vector<bool> ignore_mask = status_to_ignore_mask(status_pointer);
    auto _t_plan = _Clock::now();
    std::vector<STEP> steps = find_steps(selection_mask, resample_mask, ignore_mask, param);
    if (timings) timings->step_plan_ms += _ms(_t_plan);

    if (steps.size() == 0) {
        if (timings) timings->total_gen_ms += _ms(_t_total);
        return;
    }

    // get order and reverse order of tracks
    int nt = status_pointer->tracks_size();
    std::vector<int> order(nt, 0);
    std::vector<int> reverse_order = arange(nt);
    for (int track_num = 0; track_num < nt; track_num++) {
        midi::StatusTrack* st = status_pointer->mutable_tracks(track_num);
        order[st->track_id()] = track_num;
        st->set_track_id(track_num);
    }
    std::sort(reverse_order.begin(), reverse_order.end(),
        [&order](size_t i, size_t j) {return order[i] < order[j]; });
    util_protobuf::reorder_tracks(piece, order);

    for (int i=0; i<(int)steps.size(); i++) {
      if (i == (int)steps.size() - 1) {
        status_pointer->set_decode_final(true);
      } else {
        status_pointer->set_decode_final(false);
      }
      STEP step = steps[i];
      data_structures::LOGGER(data_structures::VERBOSITY_LEVEL_TRACE, data_structures::to_str("Sampling step :: decoding final = ", status_pointer->decode_final()));
      sample_step(piece, status_pointer, param, model, &step, callbacks, timings);
    }
    util_protobuf::reorder_tracks(piece, reverse_order);
    if (timings) timings->total_gen_ms += _ms(_t_total);
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
        if (bars_are_equivalent(input, output, track.track_id(), bar_num)) {
          identical_bars.push_back(std::make_tuple(track.track_id(), bar_num));
        }
      }
    }
  }
  return identical_bars;
}

// Count notes in selected bars across all resampled tracks.
int count_notes_in_selected_bars(midi::Piece* piece, midi::Status* status) {
  int total = 0;
  for (int track_num = 0; track_num < status->tracks_size(); track_num++) {
    const midi::StatusTrack& st = status->tracks(track_num);
    int tid = st.track_id();
    if (tid >= piece->tracks_size()) continue;
    const midi::Track& track = piece->tracks(tid);
    for (int bar_num = 0; bar_num < st.selected_bars_size(); bar_num++) {
      if (st.selected_bars(bar_num) && bar_num < track.bars_size()) {
        total += track.bars(bar_num).events_size();
      }
    }
  }
  return total;
}

// wrapper function that ensures novelty and non-silence
int sample_multi_attempts(midi::Piece* piece, midi::Status* status, midi::HyperParam* param, CallbackManager *callbacks, int max_attempts, SamplingTimings *timings = nullptr) {
  if (param->sampling_seed() >= 0) {
    torch::manual_seed(param->sampling_seed());
  }
  int attempts = 0;
  midi::Piece input;
  input.CopyFrom(*piece);
  midi::Piece best;
  bool have_best = false;
  while (attempts < max_attempts) {
    midi::Piece current;
    current.CopyFrom(*piece);
    sample(&current, status, param, callbacks, timings);
    std::vector<std::tuple<int,int>> identical_bars = find_identical_bars(&input, &current, status);
    int note_count = count_notes_in_selected_bars(&current, status);
    attempts++;
    if (identical_bars.size() == 0 && note_count > 0) {
      piece->CopyFrom(current);
      if (timings) timings->attempts = attempts;
      return attempts;
    }
    if (note_count > 0 && !have_best) {
      best.CopyFrom(current);
      have_best = true;
    }
    if (note_count == 0) {
      param->set_temperature( param->temperature() * 1.2f );
    }
    if (callbacks) {
      param->set_temperature( callbacks->update_temperature(param->temperature()) );
    }
  }
  if (have_best) {
    piece->CopyFrom(best);
  }
  if (timings) timings->attempts = attempts;
  return attempts;
}

// Parse JSON inputs common to both sample_multi_step variants
static void _parse_inputs(std::string piece_json, std::string status_json, std::string param_json,
                           midi::Piece &piece, midi::Status &status, midi::HyperParam &hyperParam) {
  util_protobuf::string_to_protobuf(piece_json, &piece);
  util_protobuf::string_to_protobuf(status_json, &status);
  util_protobuf::string_to_protobuf(param_json, &hyperParam);
  util_protobuf::validate_protobuf_fields(&piece, piece_json);
  util_protobuf::validate_protobuf_fields(&status, status_json);
  util_protobuf::validate_protobuf_fields(&hyperParam, param_json);
}

// Original API: returns (piece_json, attempts) — unchanged
std::tuple<std::string,int> sample_multi_step_py(std::string &piece_json, std::string &status_json, std::string &param_json, int max_attempts, sampling::CallbackManager *callbacks) {
  midi::Piece piece;
  midi::Status status;
  midi::HyperParam hyperParam;
  data_structures::LOGGER(data_structures::VERBOSITY_LEVEL_TRACE, "to_proto");
  _parse_inputs(piece_json, status_json, param_json, piece, status, hyperParam);
  data_structures::LOGGER(data_structures::VERBOSITY_LEVEL_VERBOSE, util_protobuf::protobuf_to_string(&status));
  data_structures::LOGGER(data_structures::VERBOSITY_LEVEL_VERBOSE, util_protobuf::protobuf_to_string(&hyperParam));
  int attempts = sample_multi_attempts(&piece, &status, &hyperParam, callbacks, max_attempts);
  return std::make_tuple(util_protobuf::protobuf_to_string(&piece), attempts);
}

// Returns the raw prompt token sequences the model would see for each step,
// without running inference.  Used by parity tests to verify that the orig
// and refactored engines feed identical token sequences to the model.
// metadata_json: the "metadata.json" extra-file embedded in the TorchScript
//                model (obtained via torch.jit.load _extra_files).
std::vector<std::vector<int>> get_infill_prompts_py(
    std::string piece_json, std::string status_json, std::string param_json,
    const std::string &metadata_json
) {
  midi::Piece piece; midi::Status status; midi::HyperParam hyperParam;
  _parse_inputs(piece_json, status_json, param_json, piece, status, hyperParam);

  midi::ModelMetadata meta;
  google::protobuf::util::JsonStringToMessage(metadata_json.c_str(), &meta);

  std::unique_ptr<encoder::ENCODER> enc = enums::getEncoderFromString(meta.encoder());
  if (!enc.get()) throw std::invalid_argument("INVALID ENCODER");
  piece.set_resolution(enc->config->resolution);
  hyperParam.set_internal_skip_preprocess(true);
  hyperParam.set_batch_size(1);

  midi::Status status_object(status);
  midi::Status *sp = &status_object;

  util_protobuf::validate_inputs(&piece, sp, &hyperParam);
  util_protobuf::pad_piece_with_status(&piece, sp, hyperParam.model_dim());
  add_timesigs_to_status(&piece, sp);
  apply_future_flags_from_status(&piece, sp);
  override_piece_features(&piece, sp, enc->rep);

  std::vector<std::vector<bool>> sel = status_to_selection_mask(sp);
  if (!any(sel)) return {};
  std::vector<bool> resample_mask = status_to_resample_mask(sp);
  std::vector<bool> ignore_mask   = status_to_ignore_mask(sp);
  std::vector<STEP> steps = find_steps(sel, resample_mask, ignore_mask, &hyperParam);
  if (steps.empty()) return {};

  // Reorder tracks as sample() does
  int nt = sp->tracks_size();
  std::vector<int> order(nt, 0);
  for (int ti = 0; ti < nt; ti++) {
    midi::StatusTrack *st = sp->mutable_tracks(ti);
    order[st->track_id()] = ti;
    st->set_track_id(ti);
  }
  util_protobuf::reorder_tracks(&piece, order);

  // For each step, build prompt via SAMPLE_CONTROL (no model needed)
  std::vector<std::vector<int>> all_prompts;
  for (const auto &s : steps) {
    midi::Piece step_piece   = piece_subset(&piece, s.start, s.end, s.get_tracks());
    midi::Status step_status = status_subset(sp, s.start, s.end, s.get_tracks());
    status_rehighlight(&step_status, s.get_bars_to_generate());
    SAMPLE_CONTROL sc(&step_piece, &step_status, &hyperParam, &meta);
    all_prompts.push_back(sc.prompt);
  }
  return all_prompts;
}

// Timed API: returns (piece_json, attempts, timings_json)
std::tuple<std::string,int,std::string> sample_multi_step_timed_py(std::string &piece_json, std::string &status_json, std::string &param_json, int max_attempts, sampling::CallbackManager *callbacks) {
  midi::Piece piece;
  midi::Status status;
  midi::HyperParam hyperParam;
  _parse_inputs(piece_json, status_json, param_json, piece, status, hyperParam);
  SamplingTimings timings;
  int attempts = sample_multi_attempts(&piece, &status, &hyperParam, callbacks, max_attempts, &timings);
  return std::make_tuple(util_protobuf::protobuf_to_string(&piece), attempts, timings.to_json());
}

std::vector<std::vector<std::vector<int>>> get_step_grids(midi::Status* status, midi::HyperParam* param) {
    std::vector<std::vector<bool>> sel = status_to_selection_mask(status);
    std::vector<bool> resample_mask = status_to_resample_mask(status);
    std::vector<bool> ignore_mask = status_to_ignore_mask(status);
    std::vector<STEP> steps = find_steps(sel, resample_mask, ignore_mask, param);

    int nt = status->tracks_size();
    if (nt == 0) return {};
    int nb = status->tracks(0).selected_bars_size();

    std::vector<std::vector<std::vector<int>>> grids;
    for (const auto &s : steps) {
        std::vector<std::vector<int>> grid(nt, std::vector<int>(nb, 0));
        for (int i=0; i<nt; i++) {
            for (int j=0; j<nb; j++) {
                if (i < (int)s.step.size() && j < (int)s.step[0].size() && s.step[i][j]) {
                    grid[i][j] = 2; // GENERATION
                } else if (i < (int)s.context.size() && j < (int)s.context[0].size() && s.context[i][j]) {
                    grid[i][j] = 1; // CONTEXT
                } else if (j >= s.start && j < s.end) {
                    grid[i][j] = 3; // MASKED (In window but not context/gen)
                } else {
                    grid[i][j] = 0; // NONE
                }
            }
        }
        grids.push_back(grid);
    }
    return grids;
}

}
