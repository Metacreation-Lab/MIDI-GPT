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
    ${CMAKE_CURRENT_SOURCE_DIR}/libraries/midifile/include
    ${CMAKE_CURRENT_SOURCE_DIR}/libraries/protobuf/include
)

target_compile_definitions(midigpt_core
    PUBLIC $<$<BOOL:${MIDIGPT_NO_TORCH}>:NO_TORCH>
)

target_link_libraries(midigpt_core PUBLIC midigpt_proto midifile)

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

# ── midigpt : pybind11 Python module ──────────────────────────────────────────
pybind11_add_module(midigpt MODULE src/cpp/bindings/lib.cpp)
target_link_libraries(midigpt PRIVATE midigpt_core)

if(NOT WIN32)
    target_compile_options(midigpt PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Install the Python extension to the site-packages root so `import midigpt` works.
install(TARGETS midigpt DESTINATION .)

# ── midigpt_tracer : optional function-level instrumentation ──────────────────
if(MIDIGPT_TRACE)
    add_library(midigpt_tracer STATIC src/trace.cpp)
    target_link_libraries(midigpt PRIVATE midigpt_tracer)
    target_compile_options(midigpt PRIVATE -finstrument-functions)
endif()
