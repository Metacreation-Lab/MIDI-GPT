cmake_minimum_required(VERSION 3.21)

project(midigpt_proto)

set(PROTO_DEF
    src/enum.proto
    src/midi.proto
    src/midi_internal.proto
    src/track_type.proto
    src/feature_extraction.proto
)

# Use CONFIG mode to pick up the protobuf package config file, which chains to
# abseil and utf8_range (required for protobuf >= 3.22 / v4.x / 24.x).
# MODULE_COMPATIBLE makes the config include protobuf-module.cmake, which
# provides the protobuf_generate_cpp() helper used below.
# If the config file isn't found, fall back to the old-style finder.
set(protobuf_MODULE_COMPATIBLE ON CACHE BOOL "" FORCE)
find_package(protobuf CONFIG QUIET)
if(NOT protobuf_FOUND)
    find_package(Protobuf REQUIRED)
endif()

protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS
    ${PROTO_DEF}
    PROTOC_OUT_DIR ${CMAKE_CURRENT_BINARY_DIR}
)

add_library(midigpt_proto STATIC
    ${PROTO_SRCS}
    ${PROTO_HDRS}
)

target_include_directories(midigpt_proto PUBLIC
    ${CMAKE_CURRENT_BINARY_DIR}   # generated .pb.h files
)

# Link against the modern import target when available (handles abseil
# transitive deps correctly); fall back to the legacy variable otherwise.
if(TARGET protobuf::libprotobuf)
    target_link_libraries(midigpt_proto PUBLIC protobuf::libprotobuf)
else()
    target_include_directories(midigpt_proto PUBLIC ${Protobuf_INCLUDE_DIRS})
    target_link_libraries(midigpt_proto PUBLIC ${Protobuf_LIBRARIES})
endif()
