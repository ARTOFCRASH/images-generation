cmake_minimum_required(VERSION 3.15 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
project(pcl-matrix_transform)

find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (main main.cpp)
target_link_libraries (main ${PCL_LIBRARIES})
target_link_libraries(main ${OpenCV_LIBS})