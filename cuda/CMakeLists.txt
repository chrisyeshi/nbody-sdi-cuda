find_package (CUDA REQUIRED)

set (SOURCES cudaglbuffer.cu cudabuffer.cu nbody.cu)

cuda_add_library (nbody SHARED ${SOURCES})
target_link_libraries (nbody ${CUDA_LIBRARIES})