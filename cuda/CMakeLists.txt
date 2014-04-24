find_package (CUDA REQUIRED)

set (CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_20,code=sm_20)

set (SOURCES cudaglbuffer.cu cudabuffer.cu nbody.cu jumpflood.cu delaunay.cu)

cuda_add_library (nbody SHARED ${SOURCES})
target_link_libraries (nbody ${CUDA_LIBRARIES})