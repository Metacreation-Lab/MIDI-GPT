# ── Protobuf ──────────────────────────────────────────────────────────────────
# Requires protobuf development headers and protoc compiler.
# On Fir/Alliance HPC:  module load protobuf
# On Ubuntu/Debian:     apt install libprotobuf-dev protobuf-compiler
# On macOS:             brew install protobuf
find_package(Protobuf REQUIRED)
add_subdirectory(libraries/protobuf)   # → target: midigpt_proto

# ── midifile (git submodule) ──────────────────────────────────────────────────
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/libraries/midifile/CMakeLists.txt")
    message(FATAL_ERROR
        "midifile submodule is missing.\n"
        "Run: git submodule update --init --recursive")
endif()
set(BUILD_PROGRAMS OFF CACHE BOOL "Build midifile CLI tools" FORCE)
add_subdirectory(libraries/midifile)   # → target: midifile

# ── pybind11 ──────────────────────────────────────────────────────────────────
# Prefer pip-installed pybind11 (present when using scikit-build-core or after
# `pip install pybind11`). Fall back to the git submodule for manual CMake builds.
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/libraries/pybind11/CMakeLists.txt")
        add_subdirectory(libraries/pybind11)
    else()
        message(FATAL_ERROR
            "pybind11 not found via find_package and the submodule is missing.\n"
            "  pip install pybind11\n"
            "  OR: git submodule update --init --recursive")
    endif()
endif()

# ── Torch (optional) ──────────────────────────────────────────────────────────
# Skip entirely when MIDIGPT_NO_TORCH=ON (dataset-creation / training-only builds).
if(NOT MIDIGPT_NO_TORCH)
    if(NOT CMAKE_PREFIX_PATH)
        # Auto-detect from the active Python environment.
        execute_process(
            COMMAND "${Python3_EXECUTABLE}" -c
                    "import torch; print(torch.utils.cmake_prefix_path)"
            OUTPUT_VARIABLE _torch_cmake_prefix
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        if(_torch_cmake_prefix)
            list(APPEND CMAKE_PREFIX_PATH "${_torch_cmake_prefix}")
        elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/libraries/libtorch")
            # Fall back to a manually downloaded LibTorch.
            list(APPEND CMAKE_PREFIX_PATH
                 "${CMAKE_CURRENT_SOURCE_DIR}/libraries/libtorch")
        else()
            message(FATAL_ERROR
                "PyTorch / LibTorch not found. Options:\n"
                "  1. pip install torch                            (recommended)\n"
                "  2. cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch  (manual)\n"
                "  3. cmake -DMIDIGPT_NO_TORCH=ON                  (no inference)")
        endif()
    endif()

    find_package(Torch REQUIRED)

    # Required to avoid a symbol conflict when torch_python is loaded alongside
    # the pybind11 module. See https://github.com/pytorch/pytorch/issues/38122
    find_library(TORCH_PYTHON_LIBRARY torch_python
        PATHS "${TORCH_INSTALL_PREFIX}/lib"
        NO_DEFAULT_PATH
    )
endif()
