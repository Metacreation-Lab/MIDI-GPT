cmake_minimum_required(VERSION 3.8)

SET(CMAKE_CXX_STANDARD 20)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(CMAKE_POSITION_INDEPENDENT_CODE ON)
#we add the following line to fix a linkage issue between torch and midifile
#https://stackoverflow.com/questions/68922557/c-linker-error-undefined-reference-when-linking-package-libtorch-and-shared
#add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)

project(midigpt)

option(compute_canada "Build for Compute Canada" OFF)
option(mac_os "Build for Mac OS" OFF)
option(no_torch "No Torch" OFF)
option(no_pybind "No Pybind" OFF)
option(trace "Trace" OFF)

#Find the necessary packages to be able to link the libraries correctly
find_package(Protobuf REQUIRED)
include_directories(${Protobuf_INCLUDE_DIRS})

if(no_torch)
    add_definitions(-DNO_TORCH)
endif()

if(NOT no_torch)
	if(mac_os)
	  message("USING PYTHON PYTORCH INSTEAD")
	else()
	  set(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/libraries/libtorch/")
	endif()
	find_package(Torch REQUIRED)

	# This is necessary to avoid a symbol linkage error https://github.com/pytorch/pytorch/issues/38122 
	# https://github.com/DeepVAC/libdeepvac/blob/master/python/CMakeLists.txt
	find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")
endif()

if(compute_canada)
  include_directories("/cvmfs/soft.computecanada.ca/easybuild/software/2020/avx512/Core/python/3.8.2/include/python3.8")
endif()

#Add the directories of libraries so the project can CMake them too
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/libraries/protobuf)
if(NOT no_torch)
	add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/libraries/torch)
endif()
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/libraries/pybind11)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/libraries/midifile)

#https://stackoverflow.com/questions/8934295/add-source-in-a-subdirectory-to-a-cmake-project/54285898#54285898 
#https://crascit.com/2016/01/31/enhanced-source-file-handling-with-target_sources/


set(SRCS
	src/common/data_structures/train_config.cpp
	src/dataset_creation/compression/lz4.c
	src/dataset_creation/dataset_manipulation/bytes_to_file.cpp
	src/common/encoder/encoder_all.h
	src/lib.cpp
)
PYBIND11_ADD_MODULE(midigpt ${SRCS})

#Adding include folders of libraries to our target so we can reference them with #include
#Add subdirectory adds those to main project so they can be CMAKEd. Include dirs allows us to reference functions in main.
target_include_directories(midigpt PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/libraries/protobuf/include)
if (NOT no_torch)
	target_include_directories(midigpt PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/libraries/torch/include)
endif()
target_include_directories(midigpt PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/libraries/midifile/include)

#Linking all the libraries
target_link_libraries(midigpt PRIVATE midigpt_proto) #Our protobuf custom library
target_link_libraries(midigpt PRIVATE midifile)
if (NOT no_torch)
	target_link_libraries(midigpt PRIVATE midigpt_torch) #Our torch custom library
	#This is necessary to avoid a symbol linkage error https://github.com/pytorch/pytorch/issues/38122 
	target_link_libraries(midigpt PRIVATE "${TORCH_LIBRARIES}" ${TORCH_PYTHON_LIBRARY})
endif()

if (trace)
	add_library(tracer STATIC src/trace.cpp)
	target_link_libraries(midigpt PRIVATE tracer)
	target_compile_options(midigpt PRIVATE -Wall -Wextra -Wpedantic -finstrument-functions)
elseif(NOT WIN32)
	target_compile_options(midigpt PRIVATE -Wall -Wextra -Wpedantic)
endif()
