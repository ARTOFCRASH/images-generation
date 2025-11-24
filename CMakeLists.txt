cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(image-generation)

find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(VTK 9.1 REQUIRED NO_MODULE)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (main main.cpp)
target_link_libraries (main ${PCL_LIBRARIES})
target_link_libraries(main ${OpenCV_LIBS})
