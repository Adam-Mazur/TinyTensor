# To compile:
# cd build
# cmake ..
# make
# And to run tests
# ctest
cmake_minimum_required(VERSION 3.10)
project(PyTorchImplementation)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add the main library (removed to fix the template errors)
# add_library(tensor STATIC src/tensor.cpp)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(test_tensor tests/test_tensor.cpp)
target_compile_options(test_tensor PRIVATE -g -O0)

include(FetchContent)
FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v3.4.0
)
FetchContent_MakeAvailable(Catch2)

target_link_libraries(test_tensor Catch2::Catch2WithMain)

enable_testing()

add_test(NAME TensorTests COMMAND test_tensor)