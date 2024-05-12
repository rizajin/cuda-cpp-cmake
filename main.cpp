#include <stdexcept>
#include "kernel.cu"

int main()
{
    if (!ExecuteCuda()) {
        throw new std::runtime_error("Failed with execute unit for CUDA");
    }

    return 0;
}