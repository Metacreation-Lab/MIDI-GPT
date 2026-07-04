#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../core/score.h"
#include "../core/types.h"
#include "../io/midi_reader.h"
#include "../io/midi_writer.h"
#include "../masking/attribute_value_constraint.h"
#include "../masking/bar_attribute_value_constraint.h"
#include "../masking/constraint.h"
#include "../masking/constraint_graph.h"
#include "../masking/density_constraint.h"
#include "../masking/grammar_constraint.h"
#include "../masking/polyphony_constraint.h"
#include "../sampling/generation_step.h"
#include "../sampling/selection_mask.h"
#include "../sampling/session_state.h"
#include "../sampling/step_planner.h"
#include "../tokenizer/decoder.h"
#include "../tokenizer/encoder.h"
#include "../tokenizer/domain_transforms.h"
#include "../tokenizer/encoder_config.h"
#include "../tokenizer/vocabulary.h"

#include "../core/logging.h"

namespace py = pybind11;
using namespace midigpt;
using namespace midigpt::io;
using namespace midigpt::tokenizer;
using namespace midigpt::masking;
using namespace midigpt::sampling;

PYBIND11_MODULE(_core, m) {

  // Logging
  py::enum_<LogLevel>(m, "LogLevel")
      .value("OFF", LogLevel::OFF)
      .value("ERROR", LogLevel::ERROR)
      .value("WARNING", LogLevel::WARNING)
      .value("INFO", LogLevel::INFO)
      .value("DEBUG", LogLevel::DEBUG)
      .value("TRACE", LogLevel::TRACE);

  m.def("set_verbosity",
        [](int level) { Logger::set_level(static_cast<LogLevel>(level)); });

  m.def("set_verbosity", [](LogLevel level) { Logger::set_level(level); });

  // enums
  // Fully qualify TokenType: Windows SDK <winnt.h> exposes a global
  // enumerator named TokenType (in _TOKEN_INFORMATION_CLASS) which would
  // shadow midigpt::TokenType in template-argument lookup under MSVC.
  py::enum_<::midigpt::TokenType>(m, "TokenType")
      .value("PieceStart", TokenType::PieceStart)
      .value("NoteOnset", TokenType::NoteOnset)
      .value("NoteOffset", TokenType::NoteOffset)
      .value("NotePitch", TokenType::NotePitch)
      .value("NonPitch", TokenType::NonPitch)
      .value("Velocity", TokenType::Velocity)
      .value("TimeDelta", TokenType::TimeDelta)
      .value("TimeAbsolutePos", TokenType::TimeAbsolutePos)
      .value("Instrument", TokenType::Instrument)
      .value("Bar", TokenType::Bar)
      .value("BarEnd", TokenType::BarEnd)
      .value("Track", TokenType::Track)
      .value("TrackEnd", TokenType::TrackEnd)
      .value("DrumTrack", TokenType::DrumTrack)
      .value("FillIn", TokenType::FillIn)
      .value("FillInPlaceholder", TokenType::FillInPlaceholder)
      .value("FillInStart", TokenType::FillInStart)
      .value("FillInEnd", TokenType::FillInEnd)
      .value("Header", TokenType::Header)
      .value("VelocityLevel", TokenType::VelocityLevel)
      .value("Genre", TokenType::Genre)
      .value("NoteDensity", TokenType::NoteDensity)
      .value("TimeSig", TokenType::TimeSig)
      .value("Segment", TokenType::Segment)
      .value("SegmentEnd", TokenType::SegmentEnd)
      .value("SegmentFillIn", TokenType::SegmentFillIn)
      .value("NoteDuration", TokenType::NoteDuration)
      .value("AvPolyphony", TokenType::AvPolyphony)
      .value("MinPolyphony", TokenType::MinPolyphony)
      .value("MaxPolyphony", TokenType::MaxPolyphony)
      .value("MinNoteDuration", TokenType::MinNoteDuration)
      .value("MaxNoteDuration", TokenType::MaxNoteDuration)
      .value("NumBars", TokenType::NumBars)
      .value("MinPolyphonyHard", TokenType::MinPolyphonyHard)
      .value("MaxPolyphonyHard", TokenType::MaxPolyphonyHard)
      .value("MinNoteDurationHard", TokenType::MinNoteDurationHard)
      .value("MaxNoteDurationHard", TokenType::MaxNoteDurationHard)
      .value("RestPercentage", TokenType::RestPercentage)
      .value("PitchClass", TokenType::PitchClass)
      .value("PitchClassCount", TokenType::PitchClassCount)
      .value("BarLevelOnsetDensity", TokenType::BarLevelOnsetDensity)
      .value("BarLevelOnsetPolyphonyMin", TokenType::BarLevelOnsetPolyphonyMin)
      .value("BarLevelOnsetPolyphonyMax", TokenType::BarLevelOnsetPolyphonyMax)
      .value("TrackLevelOnsetDensity", TokenType::TrackLevelOnsetDensity)
      .value("TrackLevelOnsetPolyphonyMin",
             TokenType::TrackLevelOnsetPolyphonyMin)
      .value("TrackLevelOnsetPolyphonyMax",
             TokenType::TrackLevelOnsetPolyphonyMax)
      .value("TrackLevelOnsetDensityMin", TokenType::TrackLevelOnsetDensityMin)
      .value("TrackLevelOnsetDensityMax", TokenType::TrackLevelOnsetDensityMax)
      .value("TrackLevelPitchRangeMin", TokenType::TrackLevelPitchRangeMin)
      .value("TrackLevelPitchRangeMax", TokenType::TrackLevelPitchRangeMax)
      .value("KeySignature", TokenType::KeySignature)
      .value("BarLevelPitchClassSet", TokenType::BarLevelPitchClassSet)
      .value("TrackLevelSilenceProportionMin",
             TokenType::TrackLevelSilenceProportionMin)
      .value("TrackLevelSilenceProportionMax",
             TokenType::TrackLevelSilenceProportionMax)
      .value("ValenceSpotify", TokenType::ValenceSpotify)
      .value("EnergySpotify", TokenType::EnergySpotify)
      .value("DanceabilitySpotify", TokenType::DanceabilitySpotify)
      .value("Danceability", TokenType::Danceability)
      .value("Tension", TokenType::Tension)
      .value("ContainsNoteDurationThirtySecond",
             TokenType::ContainsNoteDurationThirtySecond)
      .value("ContainsNoteDurationSixteenth",
             TokenType::ContainsNoteDurationSixteenth)
      .value("ContainsNoteDurationEighth",
             TokenType::ContainsNoteDurationEighth)
      .value("ContainsNoteDurationQuarter",
             TokenType::ContainsNoteDurationQuarter)
      .value("ContainsNoteDurationHalf", TokenType::ContainsNoteDurationHalf)
      .value("ContainsNoteDurationWhole", TokenType::ContainsNoteDurationWhole)
      .value("WnbdSyncopation", TokenType::WnbdSyncopation)
      .value("Repetition", TokenType::Repetition)
      .value("Delta", TokenType::Delta)
      .value("DeltaDirection", TokenType::DeltaDirection)
      .value("None", TokenType::None)
      .value("MaskBar", TokenType::MaskBar)
      .value("TensionDrum", TokenType::TensionDrum)
      .value("OnsetPolyphony", TokenType::OnsetPolyphony)
      .value("PitchRange", TokenType::PitchRange)
      .value("NoteDurationDist", TokenType::NoteDurationDist)
      .value("SilenceProportion", TokenType::SilenceProportion)
      .value("PitchClassSet", TokenType::PitchClassSet)
      .value("PieceEnd", TokenType::PieceEnd)
      .value("UseVelocity", TokenType::UseVelocity)
      .value("UseMicrotiming", TokenType::UseMicrotiming)
      .value("TrackLevelNomml", TokenType::TrackLevelNomml);

  py::enum_<TrackType>(m, "TrackType")
      .value("Melodic", TrackType::Melodic)
      .value("Drum", TrackType::Drum);

  py::enum_<BooleanEnum>(m, "BooleanEnum")
      .value("Any", BooleanEnum::Any)
      .value("False", BooleanEnum::False)
      .value("True", BooleanEnum::True);

  // core structs
  py::class_<Note>(m, "Note")
      .def(py::init<>())
      .def_readwrite("pitch", &Note::pitch)
      .def_readwrite("velocity", &Note::velocity)
      .def_readwrite("onset_ticks", &Note::onset_ticks)
      .def_readwrite("duration_ticks", &Note::duration_ticks)
      .def_readwrite("delta", &Note::delta);

  py::class_<Bar>(m, "Bar")
      .def(py::init<>())
      .def_readwrite("note_indices", &Bar::note_indices)
      .def_readwrite("ts_numerator", &Bar::ts_numerator)
      .def_readwrite("ts_denominator", &Bar::ts_denominator)
      .def_readwrite("beat_length", &Bar::beat_length)
      .def_readwrite("has_notes", &Bar::has_notes)
      .def_readwrite("future", &Bar::future);

  py::class_<Track>(m, "Track")
      .def(py::init<>())
      .def_readwrite("bars", &Track::bars)
      .def_readwrite("instrument", &Track::instrument)
      .def_readwrite("type", &Track::type)
      .def_readwrite("attributes", &Track::attributes);

  py::class_<GenreGrouping>(m, "GenreGrouping")
      .def(py::init<>())
      .def("encode",     &GenreGrouping::encode)
      .def("decode",     &GenreGrouping::decode)
      .def("contains",   &GenreGrouping::contains)
      .def("num_genres", &GenreGrouping::num_genres);

  py::class_<Score>(m, "Score")
      .def(py::init<>())
      .def_readwrite("tracks", &Score::tracks)
      .def_readwrite("notes", &Score::notes)
      .def_readwrite("resolution", &Score::resolution)
      .def_readwrite("tempo", &Score::tempo);

  // io
  py::class_<MidiReader>(m, "MidiReader")
      .def(py::init<int>(), py::arg("resolution") = 480)
      .def("read", &MidiReader::read)
      .def("read_bytes", &MidiReader::read_bytes);

  py::class_<MidiWriter>(m, "MidiWriter")
      .def(py::init<>())
      .def("write", &MidiWriter::write)
      .def("write_bytes", &MidiWriter::write_bytes);

  // tokenizer
  py::class_<EncoderConfig>(m, "EncoderConfig")
      .def(py::init<>())
      .def_static("from_json", &EncoderConfig::from_json)
      .def("to_json", &EncoderConfig::to_json)
      .def_readwrite("resolution",               &EncoderConfig::resolution)
      .def_readwrite("decode_resolution",        &EncoderConfig::decode_resolution)
      .def_readwrite("model_dim",                &EncoderConfig::model_dim)
      .def_readwrite("emit_delta_tokens",        &EncoderConfig::emit_delta_tokens)
      .def_readwrite("supports_infill",          &EncoderConfig::supports_infill)
      .def_readwrite("supports_mask_bar_token",  &EncoderConfig::supports_mask_bar_token)
      .def_readwrite("velocity_sticky",          &EncoderConfig::velocity_sticky)
      .def_readwrite("pitch_min",                &EncoderConfig::pitch_min)
      .def_readwrite("pitch_max",                &EncoderConfig::pitch_max)
      .def_readwrite("velocity_levels",          &EncoderConfig::velocity_levels)
      .def_readwrite("note_duration_max_beats",  &EncoderConfig::note_duration_max_beats)
      .def_readwrite("attribute_controls_json",  &EncoderConfig::attribute_controls_json)
      .def("derive_token_domains",               &EncoderConfig::derive_token_domains)
      .def("add_attribute_token_domains",        &EncoderConfig::add_attribute_token_domains)
      .def_property_readonly("genre_grouping",
          [](const EncoderConfig& c) -> py::object {
              if (c.genre_grouping) return py::cast(*c.genre_grouping);
              return py::none();
          });

  py::class_<EncodeOptions>(m, "EncodeOptions")
      .def(py::init<>())
      .def_readwrite("partial_encode_track_index",
                     &EncodeOptions::partial_encode_track_index)
      .def_readwrite("partial_encode_track_bars",
                     &EncodeOptions::partial_encode_track_bars)
      .def_readwrite("multi_fill", &EncodeOptions::multi_fill)
      .def_readwrite("window_bars", &EncodeOptions::window_bars)
      .def_readwrite("use_span_masks", &EncodeOptions::use_span_masks)
      .def_readwrite("remove_future_bars", &EncodeOptions::remove_future_bars)
      .def_readwrite("use_velocity",    &EncodeOptions::use_velocity)
      .def_readwrite("use_microtiming", &EncodeOptions::use_microtiming)
      .def_readwrite("genre",           &EncodeOptions::genre);

  py::class_<Vocabulary>(m, "Vocabulary")
      .def(py::init<const EncoderConfig &>())
      .def("encode_val",
           py::overload_cast<::midigpt::TokenType, int>(&Vocabulary::encode, py::const_))
      .def("decode", &Vocabulary::decode)
      .def("size", &Vocabulary::size)
      .def("has", &Vocabulary::has)
      .def("domain_size", &Vocabulary::domain_size)
      .def("range", &Vocabulary::range)
      .def("get_type", &Vocabulary::get_type)
      .def("is_type", &Vocabulary::is_type)
      .def("config", &Vocabulary::config, py::return_value_policy::reference);

  py::class_<EncodeResult>(m, "EncodeResult")
      .def_readonly("tokens",       &EncodeResult::tokens)
      .def_readonly("hidden_spans", &EncodeResult::hidden_spans);

  py::class_<Encoder>(m, "Encoder")
      .def(py::init<const Vocabulary &>())
      .def("encode", &Encoder::encode,
           py::arg("score"), py::arg("opts") = EncodeOptions{})
      .def("encode_full", &Encoder::encode_full,
           py::arg("score"), py::arg("opts") = EncodeOptions{});

  py::class_<Decoder>(m, "Decoder")
      .def(py::init<const Vocabulary &>())
      .def("decode", &Decoder::decode);

  // masking
  py::class_<Constraint, std::shared_ptr<Constraint>>(m, "Constraint");

  py::class_<GrammarConstraint, Constraint, std::shared_ptr<GrammarConstraint>>(
      m, "GrammarConstraint")
      .def(py::init<>())
      .def("set_mask_track_start", &GrammarConstraint::set_mask_track_start)
      .def("set_mask_track_end", &GrammarConstraint::set_mask_track_end)
      .def("set_max_bars", &GrammarConstraint::set_max_bars)
      .def("set_exact_bars", &GrammarConstraint::set_exact_bars)
      .def("set_autoregressive_mode", &GrammarConstraint::set_autoregressive_mode)
      .def("set_max_tracks", &GrammarConstraint::set_max_tracks)
      .def("set_require_notes", &GrammarConstraint::set_require_notes);

  py::class_<DensityConstraint, Constraint, std::shared_ptr<DensityConstraint>>(
      m, "DensityConstraint")
      .def(py::init<int>());

  py::class_<PolyphonyConstraint, Constraint,
             std::shared_ptr<PolyphonyConstraint>>(m, "PolyphonyConstraint")
      .def(py::init<int>());

  py::class_<AttributeValueConstraint, Constraint,
             std::shared_ptr<AttributeValueConstraint>>(
      m, "AttributeValueConstraint")
      .def(py::init<::midigpt::TokenType, int>());

  py::class_<BarAttributeValueConstraint, Constraint,
             std::shared_ptr<BarAttributeValueConstraint>>(
      m, "BarAttributeValueConstraint")
      .def(py::init<::midigpt::TokenType, int, int, int>());

  py::class_<ConstraintGraph>(m, "ConstraintGraph")
      .def(py::init<>())
      .def("add_constraint", &ConstraintGraph::add_constraint)
      .def("get_mask", &ConstraintGraph::get_mask)
      .def("step", &ConstraintGraph::step);

  // sampling
  py::class_<SelectionMask>(m, "SelectionMask")
      .def(py::init<>())
      .def_readwrite("selected", &SelectionMask::selected)
      .def_readwrite("autoregressive", &SelectionMask::autoregressive)
      .def_readwrite("ignore", &SelectionMask::ignore);

  py::class_<GenerationStep>(m, "GenerationStep")
      .def(py::init<>())
      .def_readwrite("start_bar", &GenerationStep::start_bar)
      .def_readwrite("end_bar", &GenerationStep::end_bar)
      .def_readwrite("is_autoregressive", &GenerationStep::is_autoregressive)
      .def_readwrite("track_indices", &GenerationStep::track_indices)
      .def_readwrite("bars_to_generate", &GenerationStep::bars_to_generate)
      .def_readwrite("bar_mapping", &GenerationStep::bar_mapping)
      .def_readwrite("context", &GenerationStep::context);

  py::class_<StepPlanner>(m, "StepPlanner")
      .def(py::init<const SelectionMask &, const EncoderConfig &, int, int>(),
           py::arg("mask"), py::arg("config"), py::arg("bars_per_step") = 1,
           py::arg("tracks_per_step") = 1)
      .def("plan", &StepPlanner::plan);

  py::class_<SessionState>(m, "SessionState")
      .def(
          py::init<Score, const GenerationStep &, const Vocabulary &,
                   const ConstraintGraph &, const Encoder &, const Decoder &,
                   bool, bool, int, int, int>(),
          py::arg("context"), py::arg("step"), py::arg("vocab"),
          py::arg("constraints"), py::arg("encoder"), py::arg("decoder"),
          py::arg("use_span_masks") = false,
          py::arg("remove_future_bars") = false,
          py::arg("use_velocity") = -1,
          py::arg("use_microtiming") = -1,
          py::arg("genre") = -1)
      .def("complete", &SessionState::complete)
      .def("context_tokens", &SessionState::context_tokens)
      .def("hidden_spans", &SessionState::hidden_spans)
      .def("logit_mask", &SessionState::logit_mask)
      .def("advance", &SessionState::advance)
      .def("result", &SessionState::result);
}