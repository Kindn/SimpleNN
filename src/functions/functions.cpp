/*
 * filename: functions.cpp
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    Definition of some basic functions.
 */

#include "functions/functions.hpp"

namespace snn
{
    double sigmoid_scalar(double scalar)
    {
        return 1.0 / (1.0 + std::exp(-scalar));
    }

    double relu_scalar(double scalar)
    {
        return std::max(0.0, scalar);
    }

    double leaky_relu_scalar(double scalar, double a)
    {
        if (a <= 1)
            throw illegalParameterValue("a should be larger than 1!");
        else
            return std::max(scalar / a, scalar);
    }

    Matrix_d sigmoid(const Matrix_d& input)
    {
        int rows = input.rows(), cols = input.cols();
        Matrix_d output(rows, cols);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                output(i, j) = sigmoid_scalar(input(i, j));
        
        return output;
    }

    Matrix_d relu(const Matrix_d& input)
    {
        int rows = input.rows(), cols = input.cols();
        Matrix_d output(rows, cols);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                output(i, j) = relu_scalar(input(i, j));
        
        return output;
    }

    Matrix_d leaky_relu(const Matrix_d& input, double a)
    {
        if (a <= 1)
            throw illegalParameterValue("a should be larger than 1!");
        else
        {
            int rows = input.rows(), cols = input.cols();
            Matrix_d output(rows, cols);
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    output(i, j) = leaky_relu_scalar(input(i, j), a);
            
            return output;
        }
    }

    Matrix_d softmax(const Matrix_d& input, int axis)
    {
        int rows = input.rows(), cols = input.cols();
        if (axis != 0 && axis != 1 && axis != 2)
            throw illegalParameterValue("Axis should be either 0 or 1!");
        else 
        {
            Matrix_d output(rows, cols);
            switch (axis)
            {
            case 0:    // row
                for (int r = 0; r < rows; r++)
                {
                    double max_element = output(r, 0);
                    for (int c = 0; c < cols; c++)
                        if (input(r, c) > max_element)
                            max_element = input(r, c);
                    double sum = 0;
                    for (int c = 0; c < cols; c++)
                        sum += std::exp(input(r, c) - max_element);
                    for (int c = 0; c < cols; c++)
                        output(r, c) = std::exp(input(r, c) - max_element) / sum;
                }
                break;
            
            case 1:   // column
                for (int c = 0; c < cols; c++)
                {
                    double max_element = output(0, c);
                    for (int r = 0; r < rows; r++)
                        if (input(r, c) > max_element)
                            max_element = input(r, c);
                    double sum = 0;
                    for (int r = 0; r < rows; r++)
                        sum += std::exp(input(r, c) - max_element);
                    for (int r = 0; r < rows; r++)
                        output(r, c) = std::exp(input(r, c) - max_element) / sum;
                }
                break;

            case 2:  // the whole matrix
                {
                    double max_element = output(0, 0);
                    for (int r = 0; r < rows; r++)
                        for (int c = 0; c < cols; c++)
                            if(input(r, c) > max_element)
                                max_element = input(r, c);
                    double sum = 0;
                    for (int r = 0; r < rows; r++)
                        for (int c = 0; c < cols; c++)
                            sum += std::exp(input(r, c) - max_element);
                    for (int r = 0; r < rows; r++)
                        for (int c = 0; c < cols; c++)
                            output(r, c) = std::exp(input(r, c) - max_element) / sum;

                    break;
                }
            default:
                break;
            }

            return output;
            
        }
    }

    double meanSquaredLoss(const Matrix_d& y, const Matrix_d& t)
    {
        if (y.rows() != t.rows() || y.cols() != t.cols())
            throw illegalParameterValue("Size of y should match size of t!");
        else
        {
            int rows = y.rows(), cols = y.cols();
            Matrix_d error = y - t;
            double sum = 0;
            for (int r = 0; r < rows; r++)
                for (int c = 0; c < cols; c++)
                    sum += error(r, c) * error(r, c);
            
            return sum / (double)(rows * cols);
        }
        
    }

    double crossEntropyLoss(const Matrix_d& y, const Matrix_d& t)
    {
        if (y.rows() != t.rows() || y.cols() != t.cols())
            throw illegalParameterValue("Size of y should match size of t!");
        else
        {
            int rows = y.rows(), cols = y.cols();
            double sum = 0;
            for (int r = 0; r < rows; r++)
            {
                for (int c = 0; c < cols; c++)
                {
                    sum += (-t(r, c) * std::log(y(r, c) + 1e-7));
                }
                    
            }

            return sum;
        }
    }

    Matrix_d urandom(int _rows, int _cols, double bound1, double bound2)
    {
        if (_rows < 1|| _cols < 1)
            throw illegalParameterValue();
        else
        {
            Matrix_d result(_rows, _cols);
            for (int r = 0; r < _rows; r++)
            {
                for (int c = 0; c < _cols; c++)
                {
                    //srand((int)time(0));
                    double scale = rand() / double(RAND_MAX);
                    result(r, c) = bound1 + scale * (bound2 - bound1);
                }
            }
            
            return result;
        }
        
    }

    Matrix_d nrandom(int _rows, int _cols, double mean, double sigma)
    {   // using box-muller method
        if (_rows < 1|| _cols < 1)
            throw illegalParameterValue();
        else
        {
            Matrix_d result(_rows, _cols);
            srand((int)time(0));
            for (int r = 0; r < _rows; r++)
            {
                for (int c = 0; c < _cols; c++)
                {
                    //srand((int)time(0));
                    double u = rand() / double(RAND_MAX);
                    double v = rand() / double(RAND_MAX);
                    //std::cout << u << " " << v << std::endl;
                    double z = std::sqrt(-2.0 * log(u)) * std::cos(2 * 3.14159 * v);
                    result(r, c) = mean + z * sigma;
                }
            }
            
            return result;
        }
    }

    Matrix_d im2col(const Matrix_d& src, int _img_rows, int _img_cols,
                                        int _ker_rows, int _ker_cols, 
                                        int _channels, 
                                        int _row_strides, int _col_strides, 
                                        int _row_pads, int _col_pads)
    {
        const int src_rows = src.rows(), src_cols = src.cols();
        const int output_rows = (_img_rows + 2 * _row_pads - _ker_rows) / _row_strides + 1;
        const int output_cols = (_img_cols + 2 * _col_pads - _ker_cols) / _col_strides + 1;
        const int rows_per_img = output_rows * output_cols;
        const int cols_per_channel = _ker_rows * _ker_cols;

        const int batch_size = src_cols / _channels;
        const int rows = batch_size * rows_per_img;
        const int cols = cols_per_channel * _channels;
        Matrix_d result(rows, cols);
        for (int batch_ind = 0; batch_ind < batch_size; batch_ind++)
        {
            for (int index_in_output = 0; index_in_output < rows_per_img; 
                                            index_in_output++)
            {
                for (int c = 0; c < cols; c++)
                {
                    int curr_row = batch_ind * rows_per_img + index_in_output;

                    int channel_index = c / cols_per_channel; // 0, 1, 2, ... 
                    int col_in_src = batch_ind * _channels + channel_index; 
                    int row_in_output = index_in_output / output_cols;
                    int col_in_output = index_in_output % output_cols;
                    int start_row_in_img = row_in_output * _row_strides - _row_pads;
                    int start_col_in_img = col_in_output * _col_strides - _col_pads;
                    int row_in_img = start_row_in_img + (c % cols_per_channel) / _ker_cols;
                    int col_in_img = start_col_in_img + (c % cols_per_channel) % _ker_cols;

                    //std::cout << row_in_img << " " << col_in_img << " " << col_in_src << std::endl;
                    
                    if (row_in_img < 0 || row_in_img >= _img_rows
                    || col_in_img < 0 || col_in_img >= _img_cols)
                    {
                        result(curr_row, c) = 0;
                    }
                    else
                    {
                        double value = src(row_in_img * _img_cols + col_in_img, col_in_src);
                        result(curr_row, c) = value;
                    }
                }
            }
        }

        return result;
    }

    Matrix_d col2im(const Matrix_d& src, int _img_rows, int _img_cols,
                                        int _ker_rows, int _ker_cols, int _channels, 
                                        int _row_strides, int _col_strides, 
                                        int _row_pads, int _col_pads)
    {
        const int src_rows = src.rows(), src_cols = src.cols();
        const int output_rows = (_img_rows + 2 * _row_pads - _ker_rows) / _row_strides + 1;
        const int output_cols = (_img_cols + 2 * _col_pads - _ker_cols) / _col_strides + 1;
        const int rows_per_img = output_rows * output_cols;
        const int cols_per_channel = _ker_rows * _ker_cols;

        const int batch_size = src_rows / rows_per_img;
        const int rows = _img_rows * _img_cols;
        const int cols = batch_size * _channels;
        Matrix_d result(rows, cols);
        for (int batch_ind = 0; batch_ind < batch_size; batch_ind++)
        {
            for (int index_in_output = 0; index_in_output < rows_per_img; 
                                            index_in_output++)
            {
                for (int c = 0; c < src_cols; c++)
                {
                    int curr_row = batch_ind * rows_per_img + index_in_output;

                    int channel_index = c / cols_per_channel; // 0, 1, 2, ... 
                    int col_in_src = batch_ind * _channels + channel_index; 
                    int row_in_output = index_in_output / output_cols;
                    int col_in_output = index_in_output % output_cols;
                    int start_row_in_img = row_in_output * _row_strides - _row_pads;
                    int start_col_in_img = col_in_output * _col_strides - _col_pads;
                    int row_in_img = start_row_in_img + (c % cols_per_channel) / _ker_cols;
                    int col_in_img = start_col_in_img + (c % cols_per_channel) % _ker_cols;

                    //std::cout << row_in_img << " " << col_in_img << " " << col_in_src << std::endl;
                    
                    if (row_in_img < 0 || row_in_img >= _img_rows
                    || col_in_img < 0 || col_in_img >= _img_cols)
                        continue;
                    else
                        result(row_in_img * _img_cols + col_in_img, col_in_src) = src(curr_row, c);
                }
            }
        }

        return result;
    }

    Matrix_d col2im_add(const Matrix_d& src, int _img_rows, int _img_cols,
                                        int _ker_rows, int _ker_cols, int _channels, 
                                        int _row_strides, int _col_strides, 
                                        int _row_pads, int _col_pads)
    {
        const int src_rows = src.rows(), src_cols = src.cols();
        const int output_rows = (_img_rows + 2 * _row_pads - _ker_rows) / _row_strides + 1;
        const int output_cols = (_img_cols + 2 * _col_pads - _ker_cols) / _col_strides + 1;
        const int rows_per_img = output_rows * output_cols;
        const int cols_per_channel = _ker_rows * _ker_cols;

        const int batch_size = src_rows / rows_per_img;
        const int rows = _img_rows * _img_cols;
        const int cols = batch_size * _channels;
        Matrix_d result(rows, cols);
        for (int batch_ind = 0; batch_ind < batch_size; batch_ind++)
        {
            for (int index_in_output = 0; index_in_output < rows_per_img; 
                                            index_in_output++)
            {
                for (int c = 0; c < src_cols; c++)
                {
                    int curr_row = batch_ind * rows_per_img + index_in_output;

                    int channel_index = c / cols_per_channel; // 0, 1, 2, ... 
                    int col_in_src = batch_ind * _channels + channel_index; 
                    int row_in_output = index_in_output / output_cols;
                    int col_in_output = index_in_output % output_cols;
                    int start_row_in_img = row_in_output * _row_strides - _row_pads;
                    int start_col_in_img = col_in_output * _col_strides - _col_pads;
                    int row_in_img = start_row_in_img + (c % cols_per_channel) / _ker_cols;
                    int col_in_img = start_col_in_img + (c % cols_per_channel) % _ker_cols;

                    //std::cout << row_in_img << " " << col_in_img << " " << col_in_src << std::endl;
                    
                    if (row_in_img < 0 || row_in_img >= _img_rows
                    || col_in_img < 0 || col_in_img >= _img_cols)
                        continue;
                    else
                        result(row_in_img * _img_cols + col_in_img, col_in_src) += src(curr_row, c);
                }
            }
        }

        return result;
    }
}


