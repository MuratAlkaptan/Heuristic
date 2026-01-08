cmake_minimum_required(VERSION 3.16)
project(centroid_penalty CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Pagmo REQUIRED)

add_executable(centroid src/main.cpp)
target_include_directories(centroid PRIVATE include)
target_link_libraries(centroid PRIVATE Pagmo::pagmo)
