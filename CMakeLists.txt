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
enable_testing()

function(midigpt_add_test name)
    add_executable(${name} tests/cpp/${name}.cpp)
    target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests/cpp)
    target_link_libraries(${name} PRIVATE midigpt_core)
    add_test(NAME ${name} COMMAND ${name})
endfunction()

midigpt_add_test(test_score)
midigpt_add_test(test_io)
midigpt_add_test(test_vocabulary)
midigpt_add_test(test_tokenizer)
midigpt_add_test(test_constraints)
midigpt_add_test(test_step_planner)
midigpt_add_test(test_session_state)
midigpt_add_test(test_domain_transforms)

pybind11_add_module(_core MODULE src/cpp/bindings/lib.cpp)
target_link_libraries(_core PRIVATE midigpt_core)

# Copy extension next to __init__.py for in-tree pytest
add_custom_command(TARGET _core POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "$<TARGET_FILE:_core>"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/python/midigpt/$<TARGET_FILE_NAME:_core>")

# Install rule so scikit-build-core packages _core into the wheel
install(TARGETS _core LIBRARY DESTINATION midigpt)
