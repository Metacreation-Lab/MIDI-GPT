include(FetchContent)

FetchContent_Declare(pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.13.6)
FetchContent_MakeAvailable(pybind11)

FetchContent_Declare(symusic
    GIT_REPOSITORY https://github.com/Yikai-Liao/symusic.git
    GIT_TAG        v0.6.0)     # pinned — update deliberately

FetchContent_GetProperties(symusic)
if(NOT symusic_POPULATED)
    FetchContent_Populate(symusic)
    # Patch zpp.cpp to avoid C++20 structured binding compilation error in zpp_bits on Apple Clang/Clang 16+
    file(WRITE "${symusic_SOURCE_DIR}/src/io/zpp.cpp" "// Patched by MIDI-GPT to avoid structured binding compilation errors in zpp_bits\n")
    add_subdirectory(${symusic_SOURCE_DIR} ${symusic_BINARY_DIR})
endif()

target_include_directories(symusic INTERFACE
    ${symusic_SOURCE_DIR}/include
    ${symusic_SOURCE_DIR}/3rdparty
    ${symusic_SOURCE_DIR}/3rdparty/pdqsort
    ${symusic_SOURCE_DIR}/3rdparty/zpp_bits
    ${symusic_SOURCE_DIR}/3rdparty/pyvec/include
)
