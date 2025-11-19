cmake_minimum_required(VERSION 4.0.2)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(MRGPU VERSION 1.0)
project(MRGPU LANGUAGES CXX CUDA)

# Enable CUDA
enable_language(CUDA)

# Set C++ standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
add_compile_options(-std=c++17)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")

# Generate compilation database compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# add an include directory
include_directories(${PROJECT_SOURCE_DIR}/include)

add_executable(loadTestCases loadTestCases.cu sectionProcessGPU.cu ./source/util.cpp)

include(CTest)
add_subdirectory(test)