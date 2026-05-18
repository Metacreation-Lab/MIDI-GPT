include(FetchContent)

FetchContent_Declare(pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.13.1)
FetchContent_MakeAvailable(pybind11)

FetchContent_Declare(symusic
    GIT_REPOSITORY https://github.com/Yikai-Liao/symusic.git
    GIT_TAG        v0.4.5)     # pinned — update deliberately
FetchContent_MakeAvailable(symusic)

target_include_directories(symusic INTERFACE
    ${symusic_SOURCE_DIR}/include
    ${symusic_SOURCE_DIR}/3rdparty
    ${symusic_SOURCE_DIR}/3rdparty/pdqsort
    ${symusic_SOURCE_DIR}/3rdparty/zpp_bits
)
