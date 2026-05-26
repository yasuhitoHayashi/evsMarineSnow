cmake_minimum_required(VERSION 3.15)
project(particle_tracker_cli LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

add_library(tracker_core STATIC
  particle_tracking.cpp
  particle_tracking.hpp
)
target_include_directories(tracker_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(tracker_cli tracker_cli.cpp)
target_link_libraries(tracker_cli PRIVATE tracker_core)
