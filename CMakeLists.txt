cmake_minimum_required(VERSION 3.10)
project(real_time_tracking)

# C++17 を有効化
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# 最適化フラグ
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")

# OpenCV を探す
find_package(OpenCV REQUIRED)
if(NOT OpenCV_FOUND)
  message(FATAL_ERROR "OpenCV not found")
endif()

include_directories(${OpenCV_INCLUDE_DIRS})

# 実行ファイルのビルド
add_executable(real_time_tracking real_time_tracking.cpp)

# OpenCV ライブラリをリンク
target_link_libraries(real_time_tracking ${OpenCV_LIBS})