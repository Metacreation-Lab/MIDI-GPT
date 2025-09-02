cmake_minimum_required(VERSION 3.16)
project(midigpt LANGUAGES CXX C)

# Modern C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Options
option(NO_TORCH "Build without PyTorch support" OFF)
option(MAC_OS "Build for macOS" OFF)
option(CUDA_AVAILABLE "CUDA is available" OFF)

# Find required packages using system package managers
find_package(PkgConfig REQUIRED)
find_package(Protobuf REQUIRED)

# Find Python
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# Find pybind11
find_package(pybind11 REQUIRED)

# Find PyTorch if not disabled
if(NOT NO_TORCH)
    find_package(Torch REQUIRED)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
endif()

# Create proto directory if it doesn't exist and move proto files there
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/proto)

# Generate protobuf files in proto/ directory
set(PROTO_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/proto)
set(PROTO_FILES
    ${PROTO_SRC_DIR}/midi.proto
)

# Check if proto files exist, if not try to copy from old location
if(NOT EXISTS ${PROTO_SRC_DIR}/midi.proto)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/libraries/protobuf/src/midi.proto)
        message(STATUS "Copying proto files from libraries/ to proto/")
        file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/libraries/protobuf/src/midi.proto 
             DESTINATION ${PROTO_SRC_DIR})
    else()
        message(FATAL_ERROR "midi.proto not found. Please ensure proto files are in proto/ directory")
    endif()
endif()

# Generate protobuf C++ files
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${PROTO_FILES})

# Handle midifile - prefer system package or use bundled
find_path(MIDIFILE_INCLUDE_DIR 
    NAMES MidiFile.h
    PATHS /usr/include /usr/local/include
    PATH_SUFFIXES midifile
)

if(MIDIFILE_INCLUDE_DIR)
    message(STATUS "Using system midifile")
    find_library(MIDIFILE_LIBRARY NAMES midifile)
    if(NOT MIDIFILE_LIBRARY)
        message(STATUS "System midifile library not found, using bundled version")
        set(USE_BUNDLED_MIDIFILE ON)
    endif()
else()
    message(STATUS "System midifile not found, using bundled version")
    set(USE_BUNDLED_MIDIFILE ON)
endif()

if(USE_BUNDLED_MIDIFILE)
    # Keep midifile as the only bundled dependency if needed
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/libraries/midifile)
        add_subdirectory(libraries/midifile)
        set(MIDIFILE_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/libraries/midifile/include)
        set(MIDIFILE_LIBRARY midifile)
    else()
        message(FATAL_ERROR "Neither system nor bundled midifile found")
    endif()
endif()

# Main library sources
set(MIDIGPT_SOURCES
    src/lib.cpp
    src/common/data_structures/train_config.cpp
    src/dataset_creation/compression/lz4.c
    src/dataset_creation/dataset_manipulation/bytes_to_file.cpp
    ${PROTO_SRCS}
)

# Create the pybind11 module
pybind11_add_module(midigpt ${MIDIGPT_SOURCES})

# Include directories
target_include_directories(midigpt PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}  # For generated protobuf headers
    ${PROTO_SRC_DIR}
    ${MIDIFILE_INCLUDE_DIR}
    ${Protobuf_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(midigpt PRIVATE
    ${Protobuf_LIBRARIES}
    ${MIDIFILE_LIBRARY}
)

# Link PyTorch if enabled
if(NOT NO_TORCH)
    target_link_libraries(midigpt PRIVATE ${TORCH_LIBRARIES})
    target_compile_definitions(midigpt PRIVATE TORCH_AVAILABLE)
else()
    target_compile_definitions(midigpt PRIVATE NO_TORCH)
endif()

# Compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(midigpt PRIVATE
        -Wall -Wextra -Wpedantic
        -fvisibility=hidden
    )
endif()

if(MAC_OS)
    target_compile_options(midigpt PRIVATE
        -stdlib=libc++
        -mmacosx-version-min=10.14
    )
endif()

# Version info
target_compile_definitions(midigpt PRIVATE
    VERSION_INFO="${PROJECT_VERSION}"
)

# Properties
set_target_properties(midigpt PROPERTIES
    CXX_VISIBILITY_PRESET "hidden"
    VISIBILITY_INLINES_HIDDEN ON
)