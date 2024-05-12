#include <stdexcept>
#include "kernel.h"

int main()
{
    if (!ExecuteCuda({1,2,3,4,5}, {10,20,30,40,50})) {
        throw new std::runtime_error("Failed with execute unit for CUDA");
    }

    return 0;
}