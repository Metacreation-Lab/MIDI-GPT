cmake_minimum_required(VERSION 3.21)
project(midigpt VERSION 0.1.0 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(cmake/dependencies.cmake)

add_library(midigpt_core STATIC
    src/cpp/io/midi_reader.cpp
    src/cpp/io/midi_writer.cpp
    src/cpp/tokenizer/encoder_config.cpp
    src/cpp/tokenizer/vocabulary.cpp
    src/cpp/tokenizer/encoder.cpp
    src/cpp/tokenizer/decoder.cpp
    src/cpp/masking/constraint_graph.cpp
    src/cpp/masking/grammar_constraint.cpp
    src/cpp/masking/polyphony_constraint.cpp
    src/cpp/masking/density_constraint.cpp
    src/cpp/sampling/step_planner.cpp
    src/cpp/sampling/session_state.cpp
)
target_include_directories(midigpt_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/include)   # nlohmann lives here
target_link_libraries(midigpt_core PUBLIC symusic)

# Tests
add_executable(test_score tests/cpp/test_score.cpp)
target_include_directories(test_score PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests/cpp)

add_executable(test_io tests/cpp/test_io.cpp)
target_include_directories(test_io PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests/cpp)
target_link_libraries(test_io PRIVATE midigpt_core)

add_executable(test_vocabulary tests/cpp/test_vocabulary.cpp)
target_include_directories(test_vocabulary PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests/cpp)
target_link_libraries(test_vocabulary PRIVATE midigpt_core)

add_executable(test_tokenizer tests/cpp/test_tokenizer.cpp)
target_include_directories(test_tokenizer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests/cpp)
target_link_libraries(test_tokenizer PRIVATE midigpt_core)

add_executable(test_constraint_graph tests/cpp/test_constraint_graph.cpp)
target_include_directories(test_constraint_graph PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests/cpp)
target_link_libraries(test_constraint_graph PRIVATE midigpt_core)

add_executable(test_step_planner tests/cpp/test_step_planner.cpp)
target_include_directories(test_step_planner PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests/cpp)
target_link_libraries(test_step_planner PRIVATE midigpt_core)

add_executable(test_session_state tests/cpp/test_session_state.cpp)
target_include_directories(test_session_state PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests/cpp)
target_link_libraries(test_session_state PRIVATE midigpt_core)

add_executable(test_domain_transforms tests/cpp/test_domain_transforms.cpp)
target_include_directories(test_domain_transforms PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests/cpp)
target_link_libraries(test_domain_transforms PRIVATE midigpt_core)

pybind11_add_module(_core MODULE src/cpp/bindings/lib.cpp)
target_link_libraries(_core PRIVATE midigpt_core)

# Copy extension next to __init__.py for in-tree pytest
add_custom_command(TARGET _core POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "$<TARGET_FILE:_core>"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/python/midigpt_refactor/$<TARGET_FILE_NAME:_core>")
