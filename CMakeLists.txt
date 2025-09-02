cmake_minimum_required(VERSION 3.16)

project(midigpt 
    LANGUAGES CXX C
    VERSION 1.0.0
    DESCRIPTION "MIDI-GPT: Generative system for computer-assisted music composition"
)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Build options
option(NO_TORCH "Build without PyTorch support" OFF)
option(ENABLE_TRACING "Enable function tracing" OFF)

# Find required packages
find_package(Protobuf REQUIRED)
find_package(PkgConfig QUIET)

# Find pybind11
find_package(pybind11 REQUIRED)

# Find PyTorch if not disabled
if(NOT NO_TORCH)
    find_package(Torch REQUIRED)
    if(NOT Torch_FOUND)
        message(FATAL_ERROR "PyTorch not found. Install with: pip install torch>=2.4.0")
    endif()
endif()

# Try to find midifile system library, fallback to bundled version
find_path(MIDIFILE_INCLUDE_DIR 
    NAMES MidiFile.h
    PATHS /usr/include /usr/local/include
    PATH_SUFFIXES midifile
)

if(NOT MIDIFILE_INCLUDE_DIR)
    message(STATUS "System midifile not found, using bundled version")
    # We'll handle this case below
    set(USE_BUNDLED_MIDIFILE ON)
else()
    message(STATUS "Found system midifile: ${MIDIFILE_INCLUDE_DIR}")
    set(USE_BUNDLED_MIDIFILE OFF)
endif()

# Generate protobuf files
set(PROTO_FILES
    proto/enum.proto
    proto/midi.proto
    proto/midi_internal.proto
    proto/track_type.proto
    proto/feature_extraction.proto
)

protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${PROTO_FILES})

# Create protobuf library
add_library(midigpt_proto STATIC ${PROTO_SRCS} ${PROTO_HDRS})
target_include_directories(midigpt_proto PUBLIC
    ${Protobuf_INCLUDE_DIRS}
    ${CMAKE_CURRENT_BINARY_DIR}  # For generated protobuf files
)
target_link_libraries(midigpt_proto ${Protobuf_LIBRARIES})

# Handle midifile dependency
if(USE_BUNDLED_MIDIFILE)
    # Include bundled midifile as external project or subdirectory
    # For now, we'll assume it's available in external/midifile
    add_subdirectory(external/midifile EXCLUDE_FROM_ALL)
    set(MIDIFILE_TARGET midifile)
    set(MIDIFILE_INCLUDE_DIRS external/midifile/include)
else()
    # Use system midifile
    find_library(MIDIFILE_LIBRARY NAMES midifile)
    if(NOT MIDIFILE_LIBRARY)
        message(FATAL_ERROR "midifile library not found")
    endif()
    add_library(midifile INTERFACE)
    target_link_libraries(midifile INTERFACE ${MIDIFILE_LIBRARY})
    target_include_directories(midifile INTERFACE ${MIDIFILE_INCLUDE_DIR})
    set(MIDIFILE_TARGET midifile)
endif()

# Source files for the main library
set(MIDIGPT_SOURCES
    src/lib.cpp
    src/lib_encoder.cpp
    src/common/data_structures/train_config.cpp
    src/dataset_creation/compression/lz4.c
    src/dataset_creation/dataset_manipulation/bytes_to_file.cpp
)

# Collect all header files for better IDE support
file(GLOB_RECURSE MIDIGPT_HEADERS
    "src/*.h"
    "src/*.hpp"
    "include/*.h"
    "include/*.hpp"
)

# Create the main Python extension module
pybind11_add_module(midigpt ${MIDIGPT_SOURCES} ${MIDIGPT_HEADERS})

# Set target properties
target_compile_definitions(midigpt PRIVATE VERSION_INFO=${PROJECT_VERSION})

if(NO_TORCH)
    target_compile_definitions(midigpt PRIVATE NO_TORCH)
endif()

# Include directories
target_include_directories(midigpt PRIVATE
    src
    include
    ${CMAKE_CURRENT_BINARY_DIR}  # For generated protobuf files
)

if(USE_BUNDLED_MIDIFILE)
    target_include_directories(midigpt PRIVATE ${MIDIFILE_INCLUDE_DIRS})
endif()

# Link libraries
target_link_libraries(midigpt PRIVATE 
    midigpt_proto
    ${MIDIFILE_TARGET}
)

if(NOT NO_TORCH)
    target_link_libraries(midigpt PRIVATE ${TORCH_LIBRARIES})
    
    # Handle the torch_python library for better Python integration
    find_library(TORCH_PYTHON_LIBRARY torch_python 
        PATHS "${TORCH_INSTALL_PREFIX}/lib"
        NO_DEFAULT_PATH
    )
    if(TORCH_PYTHON_LIBRARY)
        target_link_libraries(midigpt PRIVATE ${TORCH_PYTHON_LIBRARY})
    endif()
endif()

# Compiler-specific options
if(MSVC)
    target_compile_options(midigpt PRIVATE /W4)
else()
    target_compile_options(midigpt PRIVATE 
        -Wall -Wextra -Wpedantic
    )
endif()

# Enable tracing if requested
if(ENABLE_TRACING AND NOT WIN32)
    add_library(tracer STATIC src/trace.cpp)
    target_link_libraries(midigpt PRIVATE tracer)
    target_compile_options(midigpt PRIVATE -finstrument-functions)
endif()

# Ensure we're compatible with PyTorch's ABI
if(NOT NO_TORCH)
    target_compile_definitions(midigpt PRIVATE TORCH_EXTENSION_NAME=midigpt)
    
    # Set the same C++ ABI as PyTorch
    if(DEFINED TORCH_CXX_FLAGS)
        target_compile_options(midigpt PRIVATE ${TORCH_CXX_FLAGS})
    endif()
endif()

# Installation rules (for development installs)
install(TARGETS midigpt
    COMPONENT python
    LIBRARY DESTINATION ${PYBIND11_INSTALL_PREFIX}
    ARCHIVE DESTINATION ${PYBIND11_INSTALL_PREFIX}
    RUNTIME DESTINATION ${PYBIND11_INSTALL_PREFIX}
)