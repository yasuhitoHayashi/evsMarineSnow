cmake_minimum_required(VERSION 3.10)
project(real_time_tracking LANGUAGES CXX)

# Require C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable optimized vectorization; on Apple Silicon, default flags should suffice
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")

# Locate dependencies
find_package(SDL2 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED COMPONENTS system filesystem)

# Include directories
include_directories(
  ${SDL2_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${Boost_INCLUDE_DIRS}
)

# Define executable target
add_executable(real_time_tracking real_time_tracking.cpp)

# Link libraries
target_link_libraries(real_time_tracking PRIVATE
  SDL2::SDL2 SDL2::SDL2main
  ${OpenCV_LIBS}
  Eigen3::Eigen
  ${Boost_LIBRARIES}
  pthread
)