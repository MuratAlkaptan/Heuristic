cmake_minimum_required(VERSION 3.16)
project(heuristic CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Pagmo REQUIRED)

add_executable(heuristic
  src/main.cpp
)

target_include_directories(heuristic PRIVATE include)
target_link_libraries(heuristic PRIVATE Pagmo::pagmo)
