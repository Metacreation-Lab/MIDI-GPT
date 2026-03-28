cmake_minimum_required(VERSION 3.21)
project(midigpt VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(MIDIGPT_NO_TORCH "Exclude LibTorch (training-only / dataset builds)" OFF)
option(MIDIGPT_TRACE    "Function-level instrumentation (-finstrument-functions)" OFF)

include(cmake/dependencies.cmake)

# ── midigpt_core : pure C++ library (no Python) ───────────────────────────────
add_library(midigpt_core STATIC
    src/common/data_structures/train_config.cpp
    src/dataset_creation/compression/lz4.c
    src/dataset_creation/dataset_manipulation/bytes_to_file.cpp
)

target_include_directories(midigpt_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/libraries/protobuf/include
)

target_compile_definitions(midigpt_core
    PUBLIC $<$<BOOL:${MIDIGPT_NO_TORCH}>:NO_TORCH>
)

if(MIDIGPT_MIDI_BACKEND STREQUAL "symusic")
    target_compile_definitions(midigpt_core PUBLIC MIDIGPT_USE_SYMUSIC)
    # symusic's CMakeLists.txt uses include_directories() (directory-scoped)
    # rather than target_include_directories(symusic PUBLIC ...), so we must
    # replicate the same paths for our own targets.
    target_include_directories(midigpt_core PUBLIC
        "${symusic_SOURCE_DIR}/include"
        "${symusic_SOURCE_DIR}/3rdparty/pdqsort"
        "${symusic_SOURCE_DIR}/3rdparty/zpp_bits"
        "${symusic_SOURCE_DIR}/3rdparty"
    )
    target_link_libraries(midigpt_core PUBLIC midigpt_proto symusic)
else()
    target_include_directories(midigpt_core PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/libraries/midifile/include
    )
    target_link_libraries(midigpt_core PUBLIC midigpt_proto midifile)
endif()

if(NOT MIDIGPT_NO_TORCH)
    target_include_directories(midigpt_core PUBLIC ${TORCH_INCLUDE_DIRS})
    target_compile_options(midigpt_core PUBLIC ${TORCH_CXX_FLAGS})
    target_link_libraries(midigpt_core PUBLIC
        "${TORCH_LIBRARIES}"
        ${TORCH_PYTHON_LIBRARY}
    )
endif()

if(NOT WIN32)
    target_compile_options(midigpt_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

# ── _midigpt : pybind11 Python extension (internal name) ─────────────────────
pybind11_add_module(_midigpt MODULE src/cpp/bindings/lib.cpp)
target_link_libraries(_midigpt PRIVATE midigpt_core)

if(NOT WIN32)
    target_compile_options(_midigpt PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Install the extension inside the midigpt/ package directory so that
# `import midigpt` finds the package with __init__.py + _midigpt.so.
install(TARGETS _midigpt DESTINATION midigpt)

# ── post-build: assemble midigpt/ package in build tree for test imports ──────
# After compilation, copy __init__.py next to the built _midigpt.so so that
# pytest can do sys.path.insert(0, build_dir) + import midigpt without a full
# pip install.
add_custom_command(TARGET _midigpt POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory
        "${CMAKE_BINARY_DIR}/midigpt"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/src/python/midigpt/__init__.py"
        "${CMAKE_BINARY_DIR}/midigpt/__init__.py"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "$<TARGET_FILE:_midigpt>"
        "${CMAKE_BINARY_DIR}/midigpt/$<TARGET_FILE_NAME:_midigpt>"
    COMMENT "Assembling midigpt Python package in build tree"
)

# ── midigpt_tracer : optional function-level instrumentation ──────────────────
if(MIDIGPT_TRACE)
    add_library(midigpt_tracer STATIC src/trace.cpp)
    target_link_libraries(_midigpt PRIVATE midigpt_tracer)
    target_compile_options(_midigpt PRIVATE -finstrument-functions)
endif()
