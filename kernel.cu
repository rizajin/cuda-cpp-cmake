#include "kernel.h"
#include <stdexcept>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void addKernel(const int* a, const int* b, int* result)
{
    int i = threadIdx.x;
    result[i] = a[i] + b[i];
}

void DeAlloc(int* a, int* b, int* c)
{
    cudaError_t e{ cudaError_t::cudaSuccess };

    e = cudaFree(a);
    if (e != cudaError_t::cudaSuccess) {
        //msg

    }
    e = cudaFree(b);
    if (e != cudaError_t::cudaSuccess) {
        //msg

    }
    e = cudaFree(c);
    if (e != cudaError_t::cudaSuccess) {
        //msg

    }
}

cudaError_t addWithCuda(const std::vector<int>& aset, const std::vector<int>& bset, std::vector<int>& result)
{
    int* a = 0;
    int* b = 0;
    int* res = 0;
    std::vector<int*> cudaObjects;

    cudaError_t cudaStatus{ cudaError_t::cudaSuccess };

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaError_t::cudaSuccess)
    {
        throw std::runtime_error("cudaSetDevice Failed");
    }

    cudaStatus = cudaMalloc((void**)&a, sizeof(int) * aset.size());
    if (cudaStatus != cudaError_t::cudaSuccess)
    {
        DeAlloc(a, b, res);
        throw std::runtime_error("cudaMalloc for a failed");
    }
    cudaStatus = cudaMalloc((void**)&b, sizeof(int) * bset.size());
    if (cudaStatus != cudaError_t::cudaSuccess)
    {
        DeAlloc(a, b, res);
        throw std::runtime_error("cudaMalloc for b failed");
    }
    cudaStatus = cudaMalloc((void**)&res, sizeof(int) * result.size());
    if (cudaStatus != cudaError_t::cudaSuccess)
    {
        DeAlloc(a, b, res);
        throw std::runtime_error("cudaMalloc for res failed");
    }

    cudaStatus = cudaMemcpy(a, aset.data(), sizeof(int) * aset.size(), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaError_t::cudaSuccess)
    {
        DeAlloc(a, b, res);
        throw std::runtime_error("cudaMemcpy for a failed");
    }
    cudaStatus = cudaMemcpy(b, bset.data(), sizeof(int) * bset.size(), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaError_t::cudaSuccess)
    {
        DeAlloc(a, b, res);
        throw std::runtime_error("cudaMemcpy for a failed");
    }

    addKernel<<<1, aset.size()>>>(a, b, res);
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaError_t::cudaSuccess)
    {
        DeAlloc(a, b, res);
        std::string e = "addKernel launch failed: ";
        e.append(cudaGetErrorString(cudaStatus));
        throw std::runtime_error(e);
    }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaError_t::cudaSuccess)
    {
        DeAlloc(a, b, res);
        std::string e = "cudaDeviceSynchronize error after addKernel launch: ";
        e.append(cudaGetErrorString(cudaStatus));
        throw std::runtime_error(e);
    }

    cudaStatus = cudaMemcpy(result.data(), res, sizeof(int) * aset.size(), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaError_t::cudaSuccess)
    {
        DeAlloc(a, b, res);
        throw std::runtime_error("cudaMemcpy failed for res");
    }

    return cudaStatus;
}

void ExecuteCuda(const std::vector<int>& setA, const std::vector<int>& setB, std::vector<int>& result)
{
    result.resize(setA.size());
    cudaError_t cudaStatus = addWithCuda(setA, setB, result);
    if (cudaStatus != cudaError_t::cudaSuccess)
    {
        throw std::runtime_error("addWithCuda failed");
    }

    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaError_t::cudaSuccess)
    {
        throw std::runtime_error("cudaDeviceReset failed");
    }
}