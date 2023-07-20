cmake_minimum_required(VERSION 3.05)
project(testFace)

set(CMAKE_CXX_STANDARD 14)

set(ncnn_DIR /home/jetson/ncnn/build/install/lib/cmake/ncnn)
set(OpenCV_DIR /usr/include/opencv4)
set(OpenCV_INCLUDE_DIRS /home/jetson/app/scr/opencv/build/include)

find_package(ncnn REQUIRED)

find_package(OpenCV REQUIRED)

include_directories(./include)

file(GLOB CPP_SOURCES "./src/*.cpp" )

file(GLOB HEADERS "./include/*.h")

add_executable(${PROJECT_NAME} "main.cpp" ${CPP_SOURCES} ${HEADERS} )

target_link_libraries(${PROJECT_NAME} ncnn ${OpenCV_LIBS} -pthread)