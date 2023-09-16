/*
 * filename: common.hpp
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    Extra functions about Matrix.
 */

#ifndef  _COMMON_HPP_
#define  _COMMON_HPP_

#include <Matrix/Matrix.hpp>
#include <vector>
#include <string>
#include <fstream>

namespace snn
{
    template<class Type>
    Matrix<Type> eye(int n);

    template<class Type>
    Matrix<Type> zeros(int _rows, int _cols);

    template<class Type>
    Matrix<Type> zeros_like(const Matrix<Type>& temp);

    template<class Type>
    Matrix<Type> argmax_axis(const Matrix<Type>& src, const std::vector<int>& pos, 
                                                int axis = 0);

    template<class Type>
    Matrix<Type> sum_axis(const Matrix<Type>& src, int axis = 0);

    /********************************************************/


    template<class Type>
    Matrix<Type> eye(int n)
    {
        Matrix<Type> result(n, n);
        for (int i = 0; i < n; i++)
            result(i, i) = 1;

        return result;
    }

    template<class Type>
    Matrix<Type> zeros(int _rows, int _cols)
    {
        Matrix<Type> result(_rows, _cols);

        return result;
    }

    template<class Type>
    Matrix<Type> zeros_like(const Matrix<Type>& temp)
    {
        int rows = temp.rows(), cols = temp.cols();
        Matrix<Type> result(rows, cols);

        return result;
    }

    template<class Type>
    Matrix<Type> argmax_axis(const Matrix<Type>& src, std::vector<int>& pos, 
                                                int axis)
    {
        if (axis != 0 && axis != 1)
            throw illegalParameterValue();
        else
        {
            int rows = src.rows(), cols = src.cols();
            switch (axis)
            {
            case 0:
                {
                    pos.resize(rows);
                    pos.reserve(rows);
                    Matrix<Type> result(rows, 1);
                    for (int r = 0; r < rows; r++)
                    {
                        Type max_element = src(r, 0);
                        for (int c = 0; c < cols; c++)
                        {
                            if (src(r, c) > max_element)
                            {
                                max_element = src(r, c);
                                pos[r] = c;
                            }
                        }
                        result(r, 0) = max_element;
                    }
                    return result;
                }
            
            case 1:
                {
                    pos.resize(rows);
                    pos.reserve(rows);
                    Matrix<Type> result(1, cols);
                    for (int c = 0; c < cols; c++)
                    {
                        Type max_element = src(0, c);
                        for (int r = 0; r < rows; r++)
                        {
                            if (src(r, c) > max_element)
                            {
                                max_element = src(r, c);
                                pos[c] = r;
                            }
                        }
                        result(0, c) = max_element;
                    }
                    return result;
                }
            default:
                break;
            }
        }
    }

    template<class Type>
    Matrix<Type> sum_axis(const Matrix<Type>& src, int axis)
    {
        if (axis != 0 && axis != 1)
            throw illegalParameterValue();
        else
        {
            int rows = src.rows(), cols = src.cols();
            switch (axis)
            {
            case 0:
                {
                    Matrix_d result(rows, 1);
                    for (int r = 0; r < rows; r++)
                    {
                        double sum = 0;
                        for (int c = 0; c < cols; c++)
                            sum += src(r, c);
                        result(r, 0) = sum;
                    }
                    return result;
                }
            
            case 1:
                {
                    Matrix_d result(1, cols);
                    for (int c = 0; c < cols; c++)
                    {
                        double sum = 0;
                        for (int r = 0; r < rows; r++)
                            sum += src(r, c);
                        result(0, c) = sum;
                    }
                    return result;
                }
            default:
                break;
            }
        }
    }
}

#endif //_COMMON_HPP_
