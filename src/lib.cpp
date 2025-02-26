#pragma GCC diagnostic ignored "-Wunused-parameter"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/iostream.h>
namespace py = pybind11;

#include "common/encoder/encoder_all.h"


#include "common/midi_parsing/midi_io.h"
#include "./inference/dataset/jagged.h"
#include "./inference/enum/model_type.h"
#include "./inference/enum/encoder_types.h"
#include "./inference/sampling/control.h"
#include "./inference/sampling/callback_base.h"
#include "./inference/version.h"

#include "./common/midi_parsing/feature_extraction.h"

#ifndef NO_TORCH
#include "./inference/sampling/sample_internal.h"
#include "./inference/sampling/multi_step_sample.h"
#endif

#include <iostream>
#include <string>
#include "../include/dataset_creation/dataset_manipulation/bytes_to_file.h"
#include "../libraries/protobuf/include/proto_library.h"
#include "../libraries/torch/include/torch_library.h"
#include "../libraries/protobuf/build/midi.pb.h"
#include "MidiFile.h"
#include "./common/data_structures/train_config.h"
#include "./lib_encoder.h"

// ======================

namespace midigpt { // you can probably remove this namespace
std::string generate_py(std::string &status_str, std::string &piece_str, std::string &param_str) {
  midi::Piece piece;
  google::protobuf::util::JsonStringToMessage(piece_str.c_str(), &piece);
  midi::Status status;
  google::protobuf::util::JsonStringToMessage(status_str.c_str(), &status);
  midi::HyperParam param;
  google::protobuf::util::JsonStringToMessage(param_str.c_str(), &param);
  #ifndef NO_TORCH
  sampling::sample(&piece, &status, &param, NULL);
  #endif
 
  std::string output_str;
  google::protobuf::util::MessageToJsonString(piece, &output_str);
  return output_str;
}
}

// MAYBE THESE SHOULD GO IN A SEPARATE FILE FOR PYTHON WRAPPERS
midi::Piece string_to_piece(std::string json_string) {
  midi::Piece x;
  google::protobuf::util::JsonStringToMessage(json_string.c_str(), &x);
  return x;
}

std::string piece_to_string(midi::Piece x) {
  std::string json_string;
  google::protobuf::util::MessageToJsonString(x, &json_string);
  return json_string;
}

std::string select_random_segment_py(std::string json_string, int num_bars, int min_tracks, int max_tracks, int seed) {
  std::mt19937 engine(seed);
  midi::Piece x;
  util_protobuf::string_to_protobuf(json_string, &x);
  util_protobuf::select_random_segment(&x, num_bars, min_tracks, max_tracks, &engine);
  return util_protobuf::protobuf_to_string(&x);
}
// MAYBE THESE SHOULD GO IN A SEPARATE FILE FOR PYTHON WRAPPERS


py::bytes midi_to_json_bytes(std::string &filepath, data_structures::TrainConfig *tc, std::string &metadata_labels) {
  std::string x;
  midi::Piece p;
  auto config = std::make_shared<data_structures::EncoderConfig>();
  config->resolution = tc->resolution;
  config->decode_resolution = tc->decode_resolution;
  config->delta_resolution = tc->delta_resolution;
  config->use_microtiming = tc->use_microtiming;
  midi_io::ParseSong(filepath, &p, config);
  util_protobuf::UpdateValidSegments(&p, tc->num_bars, tc->min_tracks);
  if (!p.internal_valid_segments_size()) {
    return py::bytes(x); // empty bytes
  }

  // insert metadata labels here
  midi::MetadataLabels *ml = new midi::MetadataLabels();
  google::protobuf::util::JsonStringToMessage(metadata_labels, ml);
  p.set_allocated_internal_metadata_labels(ml);

  p.SerializeToString(&x);
  return py::bytes(x);
}

std::string json_bytes_to_string(py::bytes &json_bytes) {
  midi::Piece p;
  p.ParseFromString(json_bytes);
  return util_protobuf::protobuf_to_string(&p);
}


PYBIND11_MODULE(midigpt,handle) {

  handle.def("select_random_segment", &select_random_segment_py);
  handle.def("status_from_piece", &util_protobuf::status_from_piece_py);
  handle.def("default_sample_param", &util_protobuf::default_sample_param_py);
  handle.def("prune_tracks", &util_protobuf::prune_tracks_py);

  handle.def("version", &version);
  handle.def("getEncoderSize", &enums::getEncoderSize);
  handle.def("getEncoderType", &enums::getEncoderType);
  handle.def("getEncoder", &enums::getEncoder);
  handle.def("getEncoderTypeList", &enums::getEncoderTypeList);
  handle.def("getAttributeControlStr", &encoder::getAttributeControlStr);

#ifndef NO_TORCH
  handle.def("sample_multi_step", &sampling::sample_multi_step_py);
  handle.def("sample_multi_step_capture_output", [](std::string piece_json, std::string status_json, std::string param_json, int max_attempts, sampling::CallbackManager *callbacks) {
    py::scoped_ostream_redirect stream(
        std::cout,                               
        py::module_::import("sys").attr("stdout") // Python output
    );
    return sampling::sample_multi_step_py(piece_json, status_json, param_json, max_attempts, callbacks);
  });
  handle.def("get_notes", &sampling::get_notes_py);
#endif

  handle.def("compute_all_attribute_controls", &encoder::compute_all_attribute_controls_py);
  handle.def("get_instruments_by_category", &enums::get_instruments_by_category);
  handle.def("get_instrument_and_track_type_from_gm_inst", &enums::get_instrument_and_track_type_from_gm_inst);
  handle.def("midi_to_json_bytes", &midi_to_json_bytes);
  handle.def("json_bytes_to_string", &json_bytes_to_string);

  py::enum_<enums::MODEL_TYPE>(handle, "MODEL_TYPE", py::arithmetic())
    .value("TRACK_MODEL", enums::MODEL_TYPE::TRACK_MODEL)
    .value("BAR_INFILL_MODEL", enums::MODEL_TYPE::BAR_INFILL_MODEL)
    .export_values();

  py::class_<compression::Jagged>(handle, "Jagged")
    .def(py::init<std::string &>())
    .def("set_seed", &compression::Jagged::set_seed)
    .def("set_num_bars", &compression::Jagged::set_num_bars)
    .def("set_min_tracks", &compression::Jagged::set_min_tracks)
    .def("set_max_tracks", &compression::Jagged::set_max_tracks)
    .def("set_max_seq_len", &compression::Jagged::set_max_seq_len)
    .def("enable_write", &compression::Jagged::enable_write)
    .def("enable_read", &compression::Jagged::enable_read)
    .def("append", &compression::Jagged::append)
    .def("read", &compression::Jagged::read)
    .def("read_bytes", &compression::Jagged::read_bytes)
    .def("read_json", &compression::Jagged::read_json)
    .def("read_batch", &compression::Jagged::read_batch)
    .def("load_random_piece", &compression::Jagged::load_random_piece_py)
    .def("load_piece", &compression::Jagged::load_piece)
    .def("close", &compression::Jagged::close)
    .def("get_size", &compression::Jagged::get_size)
    .def("get_split_size", &compression::Jagged::get_split_size);

  py::class_<data_structures::TrainConfig>(handle, "TrainConfig")
    .def(py::init<>())
    .def_readwrite("num_bars", &data_structures::TrainConfig::num_bars)
    .def_readwrite("min_tracks", &data_structures::TrainConfig::min_tracks)
    .def_readwrite("max_tracks", &data_structures::TrainConfig::max_tracks)
    .def_readwrite("max_mask_percentage", &data_structures::TrainConfig::max_mask_percentage)
    .def_readwrite("no_max_length", &data_structures::TrainConfig::no_max_length)
    .def_readwrite("resolution", &data_structures::TrainConfig::resolution)
    .def_readwrite("use_microtiming", &data_structures::TrainConfig::use_microtiming)
    .def_readwrite("microtiming", &data_structures::TrainConfig::microtiming)
    .def_readwrite("decode_resolution", &data_structures::TrainConfig::decode_resolution)
    .def_readwrite("delta_resolution", &data_structures::TrainConfig::delta_resolution)
    .def("to_json", &data_structures::TrainConfig::ToJson)
    .def("from_json", &data_structures::TrainConfig::FromJson);

  py::class_<encoder::REPRESENTATION,std::shared_ptr<encoder::REPRESENTATION>>(handle, "REPRESENTATION")
    .def(py::init<std::vector<std::pair<midi::TOKEN_TYPE,encoder::TOKEN_DOMAIN>>>())
    .def("decode", &encoder::REPRESENTATION::decode)
    .def("is_token_type", &encoder::REPRESENTATION::is_token_type)
    .def("in_domain", &encoder::REPRESENTATION::in_domain)
    .def("encode", &encoder::REPRESENTATION::encode)
    .def("encode_partial", &encoder::REPRESENTATION::encode_partial_py_int)
    .def("encode_to_one_hot", &encoder::REPRESENTATION::encode_to_one_hot)
    .def("pretty", &encoder::REPRESENTATION::pretty)
    .def_readonly("vocab_size", &encoder::REPRESENTATION::vocab_size)
    .def("get_type_mask", &encoder::REPRESENTATION::get_type_mask)
    .def("max_token", &encoder::REPRESENTATION::max_token)
    .def_readonly("token_domains", &encoder::REPRESENTATION::token_domains);
  
  py::class_<encoder::TOKEN_DOMAIN>(handle, "TOKEN_DOMAIN")
    .def(py::init<int>());

py::class_<data_structures::EncoderConfig,std::shared_ptr<data_structures::EncoderConfig>>(handle, "EncoderConfig")
  .def(py::init<>())
  .def("ToJson", &data_structures::EncoderConfig::ToJson)
  .def("FromJson", &data_structures::EncoderConfig::FromJson)
  .def_readwrite("both_in_one", &data_structures::EncoderConfig::both_in_one)
  .def_readwrite("unquantized", &data_structures::EncoderConfig::unquantized)
  .def_readwrite("do_multi_fill", &data_structures::EncoderConfig::do_multi_fill)

  .def_readwrite("use_velocity_levels", &data_structures::EncoderConfig::use_velocity_levels)
  .def_readwrite("use_microtiming", &data_structures::EncoderConfig::use_microtiming)
  .def_readwrite("transpose", &data_structures::EncoderConfig::transpose)
  .def_readwrite("resolution", &data_structures::EncoderConfig::resolution)
  .def_readwrite("decode_resolution", &data_structures::EncoderConfig::decode_resolution)
  .def_readwrite("decode_final", &data_structures::EncoderConfig::decode_final)
  .def_readwrite("delta_resolution", &data_structures::EncoderConfig::delta_resolution)
  .def_readwrite("multi_fill", &data_structures::EncoderConfig::multi_fill);

py::enum_<midi::TOKEN_TYPE>(handle, "TOKEN_TYPE", py::arithmetic())
  .value("PIECE_START", midi::TOKEN_PIECE_START)
  .value("NOTE_ONSET", midi::TOKEN_NOTE_ONSET)
  .value("PITCH", midi::TOKEN_PITCH)
  .value("VELOCITY", midi::TOKEN_VELOCITY)
  .value("DELTA", midi::TOKEN_DELTA)
  .value("DELTA_DIRECTION", midi::TOKEN_DELTA_DIRECTION)
  .value("TIME_ABSOLUTE_POS", midi::TOKEN_TIME_ABSOLUTE_POS)
  .value("INSTRUMENT", midi::TOKEN_INSTRUMENT)
  .value("BAR", midi::TOKEN_BAR)
  .value("BAR_END", midi::TOKEN_BAR_END)
  .value("TRACK", midi::TOKEN_TRACK)
  .value("TRACK_END", midi::TOKEN_TRACK_END)
  .value("DRUM_TRACK", midi::TOKEN_DRUM_TRACK)
  .value("FILL_IN", midi::TOKEN_FILL_IN)
  .value("FILL_IN_PLACEHOLDER", midi::TOKEN_FILL_IN_PLACEHOLDER)
  .value("FILL_IN_START", midi::TOKEN_FILL_IN_START)
  .value("FILL_IN_END", midi::TOKEN_FILL_IN_END)
  .value("VELOCITY_LEVEL", midi::TOKEN_VELOCITY_LEVEL)
  .value("GENRE", midi::TOKEN_GENRE)
  .value("DENSITY_LEVEL", midi::TOKEN_DENSITY_LEVEL)
  .value("TIME_SIGNATURE", midi::TOKEN_TIME_SIGNATURE)
  .value("NOTE_DURATION", midi::TOKEN_NOTE_DURATION)
  .value("AV_POLYPHONY", midi::TOKEN_AV_POLYPHONY)
  .value("MIN_POLYPHONY", midi::TOKEN_MIN_POLYPHONY)
  .value("MAX_POLYPHONY", midi::TOKEN_MAX_POLYPHONY)
  .value("MIN_NOTE_DURATION", midi::TOKEN_MIN_NOTE_DURATION)
  .value("MAX_NOTE_DURATION", midi::TOKEN_MAX_NOTE_DURATION)
  .value("NUM_BARS", midi::TOKEN_NUM_BARS)
  .value("MIN_POLYPHONY_HARD", midi::TOKEN_MIN_POLYPHONY_HARD)
  .value("MAX_POLYPHONY_HARD", midi::TOKEN_MAX_POLYPHONY_HARD)
  .value("MIN_NOTE_DURATION_HARD", midi::TOKEN_MIN_NOTE_DURATION_HARD)
  .value("MAX_NOTE_DURATION_HARD", midi::TOKEN_MAX_NOTE_DURATION_HARD)
  .value("NONE", midi::TOKEN_NONE)
  .export_values();

 

// =========================================================
// =========================================================
// ENCODERS
// =========================================================
// =========================================================
init_encoders(handle);

//
// =========================================================
// =========================================================
// DATASET CREATION
// =========================================================
// =========================================================

//dataset_manipulation folder definitions
py::class_<dataset_manipulation::BytesToFile>(handle, "BytesToFile")
.def(py::init<std::string&>())
.def("append_bytes_to_file_stream", &dataset_manipulation::BytesToFile::appendBytesToFileStream)
.def("write_file", &dataset_manipulation::BytesToFile::writeFile)
.def("close", &dataset_manipulation::BytesToFile::close);

// callback wrappers
py::class_<sampling::CallbackBase, std::shared_ptr<sampling::CallbackBase>>(handle, "CallbackBase")
  .def(py::init<>())
  .def("on_bar_end", &sampling::CallbackBase::on_bar_end)
  .def("on_start", &sampling::CallbackBase::on_bar_end)
  .def("on_prediction", &sampling::CallbackBase::on_prediction);

py::class_<sampling::LogLikelihoodCallback, sampling::CallbackBase, std::shared_ptr<sampling::LogLikelihoodCallback>>(handle, "LogLikelihoodCallback")
  .def(py::init<>())
  .def_readwrite("loglik", &sampling::LogLikelihoodCallback::loglik)
  .def_readwrite("sequence_length", &sampling::LogLikelihoodCallback::sequence_length);

py::class_<sampling::RecordTokenSequenceCallback, sampling::CallbackBase, std::shared_ptr<sampling::RecordTokenSequenceCallback>>(handle, "RecordTokenSequenceCallback")
  .def(py::init<>())
  .def_readwrite("tokens", &sampling::RecordTokenSequenceCallback::tokens);

py::class_<sampling::TemperatureIncreaseCallback, sampling::CallbackBase, std::shared_ptr<sampling::TemperatureIncreaseCallback>>(handle, "TemperatureIncreaseCallback")
  .def(py::init<float,float>())
  .def_readwrite("current_temperature", &sampling::TemperatureIncreaseCallback::current_temperature);

py::class_<sampling::CallbackManager>(handle, "CallbackManager")
  .def(py::init<>())
  .def("add_callback", &sampling::CallbackManager::add_callback_ptr)
  .def("on_bar_end", &sampling::CallbackManager::on_bar_end)
  .def("on_prediction", &sampling::CallbackManager::on_prediction)
  .def("on_start", &sampling::CallbackManager::on_start);

}
