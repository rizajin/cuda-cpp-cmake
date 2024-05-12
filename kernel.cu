#include "kernel.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void addKernel(const int* a, const int* b, int* result)
{
    int i = threadIdx.x;
    result[i] = a[i] + b[i];
}



bool ExecuteCuda(const std::vector<int>& setA, const std::vector<int>& setB)
{


    return true;
}