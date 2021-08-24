#include <functions/functions.hpp>
#include <Matrix/Matrix.hpp>
#include <functions/common.hpp>
#include <iostream>

using namespace snn;

int main()
{
    Matrix_d m = nrandom(16, 6);
    
    std::cout << "m = " << std::endl;
    std::cout << m << std::endl;
    std::cout << "m(5, 2) = " << m(5, 2) << std::endl;

    int img_rows = 4, img_cols = 4;
    int ker_rows = 3, ker_cols = 3;
    int channels = 3;
    int strides = 2, pads = 1;
    Matrix_d a = im2col(m, img_rows, img_cols, 
                        ker_rows, ker_cols, channels, 
                        strides, strides, pads, pads);
    std::cout << "a = " << std::endl;
    std::cout << a << std::endl;

    Matrix_d b = col2im(a, img_rows, img_cols, 
                        ker_rows, ker_cols, channels, 
                        strides, strides, pads, pads);
    std::cout << "b = " << std::endl;
    std::cout << b << std::endl;

    return 0;
}

