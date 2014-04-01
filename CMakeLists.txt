cmake_minimum_required (VERSION 2.8.11)
project (nbody)

add_subdirectory (cuda)
include_directories (cuda)

set (CMAKE_INCLUDE_CURRENT_DIR ON)
set (CMAKE_AUTOMOC ON)

find_package (OpenGL REQUIRED)
find_package (Qt5Core REQUIRED)
find_package (Qt5Widgets REQUIRED)
find_package (Qt5OpenGL REQUIRED)
add_executable (viewer main.cpp glwidget.cpp)
qt5_use_modules (viewer Core Widgets OpenGL)
target_link_libraries (viewer nbody GL)