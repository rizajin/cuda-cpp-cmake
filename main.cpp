#include <stdexcept>
#include <vector>
#include <string>
#include <iostream>

#include "kernel.h"

int main()
{
    std::vector<int> a{ 1,2,3,4,5 };
    std::vector<int> b{ 10,20,30,40,50 };
    std::vector<int> result;
    // NOTE: expects all of the vectors to be of same size.
    ExecuteCuda(a, b, result);

    std::string msg{ "1,2,3,4,5 + 10,20,30,40,50 = " };
    for (const auto& x : result)
    {
        msg.append(std::to_string(x));
        if (x != result.back()) {
            msg.append(",");
        }
    }

    std::cout << msg << std::endl;

    return 0;
}