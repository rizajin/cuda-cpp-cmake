cmake_minimum_required(VERSION 3.21)
project(CudaCppCmake LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(CUDAToolkit REQUIRED)

add_executable(CudaCppCmake main.cpp kernel.h kernel.cu)

set_target_properties(
    CudaCppCmake
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
target_include_directories(CudaCppCmake PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(CudaCppCmake CUDA::toolkit CUDA::cudart)