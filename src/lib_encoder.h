#include <pybind11/pybind11.h>
namespace py = pybind11;

void init_encoders(py::module &handle) {

	py::enum_<enums::ENCODER_TYPE>(handle, "ENCODER_TYPE", py::arithmetic())
    .value("EXPRESSIVE_ENCODER", enums::ENCODER_TYPE::EXPRESSIVE_ENCODER)
		.value("NO_ENCODER", enums::ENCODER_TYPE::NO_ENCODER)
		.export_values();

  py::class_<encoder::ExpressiveEncoder>(handle, "ExpressiveEncoder")
    .def(py::init<>())
    .def("encode", &encoder::ExpressiveEncoder::encode)
    .def("decode", &encoder::ExpressiveEncoder::decode)
    .def("midi_to_json", &encoder::ExpressiveEncoder::midi_to_json)
    .def("midi_to_tokens", &encoder::ExpressiveEncoder::midi_to_tokens)
    .def("json_to_midi", &encoder::ExpressiveEncoder::json_to_midi)
    .def("json_track_to_midi", &encoder::ExpressiveEncoder::json_track_to_midi)
    .def("json_to_tokens", &encoder::ExpressiveEncoder::json_to_tokens)
    .def("tokens_to_json", &encoder::ExpressiveEncoder::tokens_to_json)
    .def("resample_delta_json", &encoder::ExpressiveEncoder::resample_delta_json)
    .def("tokens_to_midi", &encoder::ExpressiveEncoder::tokens_to_midi)
    .def("pretty", &encoder::ExpressiveEncoder::pretty)
    .def("vocab_size", &encoder::ExpressiveEncoder::vocab_size)
    .def("get_attribute_control_types", &encoder::ExpressiveEncoder::get_attribute_control_types)
    .def("set_scheme", &encoder::ExpressiveEncoder::set_scheme)
    .def_readonly("config", &encoder::ExpressiveEncoder::config)
    .def_readonly("rep", &encoder::ExpressiveEncoder::rep);

}
