cmake_minimum_required(VERSION 3.10)
project(real_time_tracking LANGUAGES CXX)

# C++17 を必須にし、拡張は無効化
set(CMAKE_CXX_STANDARD       17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS      OFF)

# デフォルトで Release ビルドに
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# 最適化フラグ（Release のみ有効）
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")

# OpenGL／GLEW
find_package(OpenGL REQUIRED)
# GLEW がモジュールとして見つからない場合はパッケージマネージャ等でインストールしてください
find_package(GLEW REQUIRED)
# Use imported target for GLEW if available
if (NOT TARGET GLEW::GLEW)
  message(FATAL_ERROR "GLEW::GLEW target is not available")
endif()

# OpenCV, Eigen3, Boost（必要なら）
find_package(OpenCV REQUIRED)
find_package(Eigen3   REQUIRED)  # Eigen3::Eigen
find_package(Boost    REQUIRED)  # Boost_INCLUDE_DIRS, Boost_LIBRARIES

# include ディレクトリ
include_directories(
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
)

# 実行ファイル定義
add_executable(real_time_tracking
  real_time_tracking.cpp
)

# ライブラリリンク
target_link_libraries(real_time_tracking PRIVATE
  ${OpenCV_LIBS}        # OpenCV
  Eigen3::Eigen         # Eigen（ヘッダオンリー）
  GLEW::GLEW            # GLEW
  ${OPENGL_gl_LIBRARY}  # OpenGL
  ${Boost_LIBRARIES}    # Boost
  pthread                # スレッド
)

# 実行時に DLL/so のパス問題を避けるために必要なら以下を有効化
# set_target_properties(real_time_tracking PROPERTIES
#   BUILD_WITH_INSTALL_RPATH TRUE
#   INSTALL_RPATH "$ORIGIN/../lib"
# )