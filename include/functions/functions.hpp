/*
 * filename: functions.hpp
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    Declaration of some basic funtions.
 */

#ifndef  _FUNCTIONS_HPP_
#define  _FUNCTIONS_HPP_

#include <Matrix/Matrix.hpp>
#include <cmath>
#include <ctime>
#include <algorithm>
#include <iostream>

namespace snn
{
    double sigmoid_scalar(double scalar);
    double relu_scalar(double scalar);
    double leaky_relu_scalar(double scalar, double a);

    Matrix_d sigmoid(Matrix_d& input);
    Matrix_d relu(Matrix_d& input);
    Matrix_d leaky_relu(Matrix_d& input, double a);
    Matrix_d softmax(Matrix_d& input, int axis = 0);

    double meanSquaredLoss(Matrix_d& y, Matrix_d& t);
    double crossEntropyLoss(Matrix_d& y, Matrix_d& t);

    Matrix_d urandom(int r, int c, double bound1 = 0.0, double bound2 = 1.0);
    Matrix_d nrandom(int r, int c, double mean = 0, double sigma = 1.0);


    Matrix_d im2col(Matrix_d& src, int _img_rows, int _img_cols,
                                        int _ker_rows, int _ker_cols, 
                                        int _channels, 
                                        int _row_strides, int _col_strides, 
                                        int _row_pads, int _col_pads);
    Matrix_d col2im(Matrix_d& src, int _img_rows, int _img_cols,
                                        int _ker_rows, int _ker_cols, int _channels, 
                                        int _row_strides, int _col_strides, 
                                        int _row_pads, int _col_pads);
    Matrix_d col2im_add(Matrix_d& src, int _img_rows, int _img_cols,
                                        int _ker_rows, int _ker_cols, int _channels, 
                                        int _row_strides, int _col_strides, 
                                        int _row_pads, int _col_pads);
}


                                     

#endif //_FUNCTIONS_HPP_
