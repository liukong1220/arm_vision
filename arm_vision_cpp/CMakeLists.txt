cmake_minimum_required(VERSION 3.10)
project(hbv_vision)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
find_package(fmt REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}) # 添加这一行以找到 "tools/..."

# 添加 tools 子目录
add_subdirectory(tools)

add_executable(calibrate_stereo calibration/calibrate_stereo.cpp)
target_link_libraries(calibrate_stereo ${OpenCV_LIBS} fmt::fmt yaml-cpp tools)

add_executable(capture calibration/capture.cpp)
target_link_libraries(capture ${OpenCV_LIBS} fmt::fmt yaml-cpp tools) # <--- 在这里添加 "yaml-cpp"

add_executable(calibrate_handeye calibration/calibrate_handeye.cpp)
target_link_libraries(calibrate_handeye ${OpenCV_LIBS} fmt::fmt yaml-cpp Eigen3::Eigen tools)

add_executable(calibrate_robotworld_handeye calibration/calibrate_robotworld_handeye.cpp)
target_link_libraries(calibrate_robotworld_handeye ${OpenCV_LIBS} fmt::fmt yaml-cpp Eigen3::Eigen tools)

add_executable(split_video calibration/split_video.cpp)
target_link_libraries(split_video ${OpenCV_LIBS} fmt::fmt tools)

add_executable(verify_stereo calibration/verify_stereo.cpp)
target_link_libraries(verify_stereo ${OpenCV_LIBS} fmt::fmt yaml-cpp)