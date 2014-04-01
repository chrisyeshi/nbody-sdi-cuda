cmake_minimum_required (VERSION 2.8.9)
project (nbody)

find_package (CUDA REQUIRED)

set (SOURCES cudabuffer.cu nbody.cu)

cuda_add_library (nbody SHARED ${SOURCES})
target_link_libraries (nbody ${CUDA_LIBRARIES})