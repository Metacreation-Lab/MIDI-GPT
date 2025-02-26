cmake_minimum_required(VERSION 3.8)

project(midigpt_proto)

set(PROTO_DEF 
	src/enum.proto
	src/midi.proto
	src/midi_internal.proto
	src/track_type.proto
	src/feature_extraction.proto)

find_package(Protobuf REQUIRED)

protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS 
	${PROTO_DEF}
	PROTOC_OUT_DIR ${CMAKE_CURRENT_BINARY_DIR} # it's the default but it does not hurt to be explicit here...
)

add_library(midigpt_proto
	${PROTO_SRCS} 
	${PROTO_HDRS})

target_include_directories(midigpt_proto
PUBLIC
	${Protobuf_INCLUDE_DIRS}
	${CMAKE_CURRENT_BINARY_DIR} # for generated protobuf files
)
target_link_libraries(midigpt_proto ${Protobuf_LIBRARIES})