cmake_minimum_required(VERSION 3.31)
project(ct_cpp25_ndarray_lehasam23)

set(CMAKE_CXX_STANDARD 23)

include_directories(lib)
find_package(GTest CONFIG REQUIRED)
add_executable(GTestCLion
        lib/NDArray.h
        test/tests.cpp)

target_link_libraries(GTestCLion PRIVATE GTest::gtest GTest::gtest_main)
