/*
 * filename: Matrix.hpp
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    Decleration of a 2D-matrix class and inplement of some
 *           frequently used matrix operations.
 */

#ifndef  _MATRIX_HPP_
#define  _MATRIX_HPP_

#include <iostream>
#include <algorithm>
#include <vector>
#include <cmath>
#include <iterator>
#include "exceptions.hpp"

template<class Type>
class Matrix;

template<class Type>
std::ostream& operator << (std::ostream& out, const Matrix<Type>& mat);

using Matrix_c    =    Matrix<char>;
using Matrix_i    =    Matrix<int>;
using Matrix_s    =    Matrix<short>;
using Matrix_l    =    Matrix<long>;
using Matrix_f    =    Matrix<float>;
using Matrix_d    =    Matrix<double>;

enum Axis
{
    ROW = 0,
    COL = 1,
    ALL = 2
};

template<class Type>
class Matrix
{
    
    friend std::ostream& operator << <>(std::ostream&, const Matrix<Type>&);

    private:
        int theRows, theCols;
        Type *data;

    public:
        Matrix();
        Matrix(int _rows, int _cols);
        Matrix(const Matrix<Type>&);
        Matrix(const Type* array, int _rows, int _cols);
        ~Matrix() {delete [] data;};

        int rows() const {return this->theRows;}
        int cols() const {return this->theCols;}
        int size() const {return this->theRows * this->theCols;}
        Type* get_data() const
        {return data;};

        template<class Out_Type>
        Matrix<Out_Type> cast() const
        {
            Matrix<Out_Type> result(this->theRows, this->theCols);
            for (int r = 0; r < result.rows(); r++)
                for (int c = 0; c < result.cols(); c++)
                    result(r, c) = (Out_Type)((*this)(r, c));

            return result;
        }
        
        virtual Type& operator () (int i, int j) const;
        virtual Matrix<Type>& operator = (const Matrix<Type>&);
        virtual Matrix<Type> operator + () const;      // unary +
        virtual Matrix<Type> operator + (const Matrix<Type>&) const;
        virtual Matrix<Type> operator + (Type) const;
        virtual Matrix<Type> operator - () const;      // unary -
        virtual Matrix<Type> operator - (const Matrix<Type>&) const;
        virtual Matrix<Type> operator - (Type) const;
        virtual Matrix<Type> operator * (const Matrix<Type>&) const;
        virtual Matrix<Type> operator * (Type) const;
        virtual Matrix<Type> operator / (const Matrix<Type>&) const;  // by element
        virtual Matrix<Type> operator / (Type) const;
        virtual Matrix<Type>& operator += (const Matrix<Type>&);
        virtual Matrix<Type>& operator -= (const Matrix<Type>&);

        virtual bool operator == (const Matrix<Type>&) const;
        virtual bool operator != (const Matrix<Type>&) const;

        virtual Matrix<Type> sub_matrix(int _row1, int _row2, int _col1, int _col2) const;
        virtual Matrix<Type> sub_matrix(std::vector<int> row_set, std::vector<int> col_set) const;
        virtual Matrix<Type> swap_row(int i, int j) const;
        virtual Matrix<Type> swap_col(int i, int j) const;
        virtual Matrix<Type> reshape(int _new_rows, int _new_cols) const;
        //virtual Matrix<Type> trans_row(int i, int j, Type c) const;
        //virtual Matrix<Type> trans_col(int i, int j, Type c) const;

        virtual bool is_null() const;
        virtual bool is_equal(const Matrix<Type>&) const;

        virtual Matrix<Type> transpose() const;
        virtual Matrix<Type> element_pow(double) const;
        //virtual Matrix<Type> eliminate() const;
        virtual int getRank() const;
        virtual Matrix<Type> cofactor(int, int) const;
        virtual Matrix<Type> inv() const;
        virtual double det() const;

        virtual double sum() const
            {
                Matrix<double> matd = this->cast<double>();
                Type s = 0;
                for (int r = 0; r < matd.rows(); r++)
                    for (int c = 0; c < matd.cols(); c++)
                        s += matd(r, c);
                
                return s;
            }
        virtual double mean() const
            {return this->sum() / (double)this->size();}

        virtual Matrix<Type> normalize(int axis = 0) const;
};

template<class Type>
std::ostream& operator << (std::ostream& out, const Matrix<Type>& mat)
{
    out << "Matrix " << mat.theRows << "x" << mat.theCols << ":" << std::endl;
    for (int i = 0; i < 2 * mat.theCols; i++)
        out << "-";
    out << std::endl;
    for (int i = 0; i < mat.theRows; i++)
    {
        for (int j = 0; j < mat.theCols; j++)
            out << mat.data[i * mat.theCols + j] << " ";
        out << std::endl;
    }
    for (int i = 0; i < 2 * mat.theCols; i++)
        out << "-";
    out << std::endl;

    return out;
}

template<class Type>
Matrix<Type>::Matrix()
{
    theRows = theCols = 0;
    data = nullptr;
}

template<class Type>
Matrix<Type>::Matrix(int _rows, int _cols)
{
    if (_rows < 0 || _cols < 0)
        throw illegalParameterValue("Both row(s) and col(s) should be 0 or positive integers!");
    else
    {
        this->theRows = _rows;
        this->theCols = _cols;
        if (theRows * theCols == 0)
            data = nullptr;
        else data = new Type [theRows * theCols]();
    }
}

template<class Type>
Matrix<Type>::Matrix(const Matrix<Type>& m)
{
    this->theRows = m.theRows;
    this->theCols = m.theCols;
    this->data = new Type [theRows * theCols]();

    std::copy(m.data, 
              m.data + this->theRows * this->theCols,
              this->data);
}

template<class Type>
Matrix<Type>::Matrix(const Type* array, int _rows, int _cols)
{
    if (_rows <= 0 || _cols <= 0)
        throw illegalParameterValue("Both row(s) and col(s) should be positive integers!");
    else
    {
        this->theRows = _rows;
        this->theCols = _cols;
        data = new Type [theRows * theCols]();
        for (int i = 0; i < theRows * theCols; i++)
            data[i] = array[i];
    }
}

template<class Type>
Type& Matrix<Type>::operator () (int i, int j) const
{
    if (i < 0 || i >= this->theRows
        || j < 0 || j >= this->theCols)
        throw matrixIndexOutOfBound();
    else
    {
        return data[i * theCols + j];
    }
}

template<class Type>
Matrix<Type>& Matrix<Type>::operator = (const Matrix<Type>& m)
{
    if (this != &m)
    {
        delete [] data;
        this->theRows = m.theRows;
        this->theCols = m.theCols;
        data = new Type [theRows * theCols];
        std::copy(m.data, 
              m.data + this->theRows * this->theCols,
              this->data);
    }

    return *this;
}

template<class Type>
Matrix<Type> Matrix<Type>::operator + () const
{
    Matrix<Type> result(theRows, theCols);
    result = *this;

    return result;
}

template<class Type>
Matrix<Type> Matrix<Type>::operator + (const Matrix<Type>& m) const
{
    if (m.theRows != this->theRows || m.theCols != this->theCols)
        throw matrixSizeMismatch();
    else
    {
        Matrix<Type> result(theRows, theCols);
        for (int i = 0; i < result.size(); i++)
        {
            result.data[i] = this->data[i] + m.data[i];
        }

        return result;
    }
}

template<class Type>
Matrix<Type> Matrix<Type>::operator + (Type scalar) const
{
        Matrix<Type> result(theRows, theCols);
        for (int i = 0; i < result.size(); i++)
        {
            result.data[i] = this->data[i] + scalar;
        }

        return result;

}

template<class Type>
Matrix<Type> Matrix<Type>::operator - () const
{
    Matrix<Type> result(theRows, theCols);
    for (int i = 0; i < result.size(); i++)
    {
        result.data[i] = -this->data[i];
    }

    return result;
}

template<class Type>
Matrix<Type> Matrix<Type>::operator - (const Matrix<Type>& m) const
{
    if (m.theRows != this->theRows || m.theCols != this->theCols)
        throw matrixSizeMismatch();
    else
    {
        Matrix<Type> result(theRows, theCols);
        for (int i = 0; i < result.size(); i++)
        {
            result.data[i] = this->data[i] - m.data[i];
        }

        return result;
    }
}

template<class Type>
Matrix<Type> Matrix<Type>::operator - (Type scalar) const
{
        Matrix<Type> result(theRows, theCols);
        for (int i = 0; i < result.size(); i++)
        {
            result.data[i] = this->data[i] - scalar;
        }

        return result;

}

template<class Type>
Matrix<Type> Matrix<Type>::operator * (const Matrix<Type>& m) const
{
    if (this->theCols != m.theRows)
        throw matrixSizeMismatch();
    else
    {
        Matrix<Type> result(this->theRows, m.theCols);
        for (int i = 0; i < result.theRows; i++)
            for (int k = 0; k < this->theCols; k++)
                for (int j = 0; j < result.theCols; j++)
                    result(i, j) += (*this)(i, k) * m(k, j);
        
        return result;
    }
}

template<class Type>
Matrix<Type> Matrix<Type>::operator * (Type scalar) const
{
    Matrix<Type> result(this->theRows, this->theCols);
    for (int i = 0; i < this->size(); i++)
        result.data[i] = this->data[i] * scalar;
    
    return result;
}

template<class Type>
Matrix<Type> Matrix<Type>::operator / (const Matrix<Type>& m) const
{
    if (m.theRows != this->theRows || m.theCols != this->theCols)
        throw matrixSizeMismatch();
    else
    {
        Matrix<Type> result(theRows, theCols);
        for (int i = 0; i < result.size(); i++)
        {
            if (m.data[i] == 0)
                throw divisionByZero();
            result.data[i] = this->data[i] / m.data[i];
        }

        return result;
    }
}

template<class Type>
Matrix<Type> Matrix<Type>::operator / (Type scalar) const
{
    Matrix<Type> result(this->theRows, this->theCols);
    for (int i = 0; i < this->size(); i++)
        result.data[i] = this->data[i] / scalar;
    
    return result;
}

template<class Type>
Matrix<Type>& Matrix<Type>::operator += (const Matrix<Type>& m)
{
    if (m.theRows != this->theRows || m.theCols != this->theCols)
        throw matrixSizeMismatch();
    else
    {
        for (int i = 0; i < this->size(); i++)
        {
            this->data[i] += m.data[i];
        }

        return *this;
    }
}

template<class Type>
Matrix<Type>& Matrix<Type>::operator -= (const Matrix<Type>& m)
{
    if (m.theRows != this->theRows || m.theCols != this->theCols)
        throw matrixSizeMismatch();
    else
    {
        for (int i = 0; i < this->size(); i++)
        {
            this->data[i] -= m.data[i];
        }

        return *this;
    }
}

template<class Type>
bool Matrix<Type>::operator == (const Matrix<Type>& m) const
{
    return this->is_equal(m);
}

template<class Type>
bool Matrix<Type>::operator != (const Matrix<Type>& m) const
{
    return !this->is_equal(m);
}

template<class Type>
Matrix<Type> Matrix<Type>::sub_matrix(int _row1, int _row2, int _col1, int _col2) const
{
    if (_row1 < 0 || _row1 >= this->theRows 
     || _row2 < 0 || _row2 >= this->theRows
     || _col1 < 0 || _col1 >= this->theCols
     || _col2 < 0 || _col2 >= this->theCols )
        throw matrixIndexOutOfBound();
    else
    {
        int sub_rows = std::abs(_row2 - _row1) + 1, sub_cols = std::abs(_col2 - _col1) + 1;
        int row_lower = std::min(_row1, _row2), row_upper = std::max(_row1, _row2);
        int col_lower = std::min(_col1, _col2), col_upper = std::max(_col1, _col2);
        Matrix<Type> result(sub_rows, sub_cols);
        int ind = 0;
        int row_step =  _row1 < _row2 ? 1 : -1, col_step = _col1 < _col2 ? 1 : -1;
        int r = _row1;
        while (r >= row_lower && r <= row_upper)
        {
            int c = _col1;
            while (c >= col_lower && c <= col_upper)
            {
                result.data[ind++] = (*this)(r, c);
                c += col_step;
            }
            r += row_step;
        }
        
        return result;
    }
}

template<class Type>
Matrix<Type> Matrix<Type>::sub_matrix(std::vector<int> row_set, std::vector<int> col_set) const
{
    if (row_set.size() <= 0 || col_set.size() <= 0)
        throw illegalParameterValue("Row set and col set should not be empty!");
    else
    {
        int sub_rows = row_set.size(), sub_cols = col_set.size();
        Matrix<Type> result(sub_rows, sub_cols);
        int ind = 0;
        for (int i = 0; i < row_set.size(); i++)
        {
            if (row_set[i] < 0 || row_set[i] >= this->theRows)
                throw matrixIndexOutOfBound();
            else
            {
                for (int j = 0; j < col_set.size(); j++)
                {
                    if (col_set[j] < 0 || col_set[j] >= this->theCols)
                        throw matrixIndexOutOfBound();
                    else
                        result.data[ind++] = (*this)(row_set[i], col_set[j]);
                }
            }
            
        }
         return result;
    }
}

template<class Type>
Matrix<Type> Matrix<Type>::swap_row(int i, int j) const
{
    if (i < 0 || i > this->theRows
     || j < 0 || j > this->theRows)
        throw matrixIndexOutOfBound();
    else
    {
        Matrix<Type> result = *this;
        for (int c = 0; c < this->theCols; c++)
        {
            Type temp = (*this)(i, c);
            (*this)(i, c) = (*this)(j, c);
            (*this)(j, c) = temp;
        }

        return result;
    }
}

template<class Type>
Matrix<Type> Matrix<Type>::swap_col(int i, int j) const
{
    if (i < 0 || i > this->theCols
     || j < 0 || j > this->theCols)
        throw matrixIndexOutOfBound();
    else
    {
        Matrix<Type> result = *this;
        for (int r = 0; r < this->theRows; r++)
        {
            Type temp = (*this)(r, i);
            (*this)(r, i) = (*this)(r, j);
            (*this)(r, j) = temp;
        }

        return result;
    }
}

template<class Type>
Matrix<Type> Matrix<Type>::reshape(int _new_rows, int _new_cols) const
{
    if (_new_rows < 1 || _new_cols < 1)
        throw illegalParameterValue();
    else
    {
        Matrix<Type> result(_new_rows, _new_cols);
        int new_size = std::min(this->size(), result.size());
        for (int i = 0; i < new_size; i++)
            result.data[i] = this->data[i];

        return result;
    }
}

template<class Type>
bool Matrix<Type>::is_null() const
{
    bool isNull = true;
    for (int i = 0; i < this->size(); i++)
    {
        if (data[i] != 0)
        {
            isNull = false;
            break;
        }
    }

    return isNull;
}

template<class Type>
bool Matrix<Type>::is_equal(const Matrix<Type>& m) const
{
    if (this->theRows == m.theRows && this->theCols == m.theCols)
    {
        for (int i = 0; i < this->size(); i++)
            if (this->data[i] != m.data[i])
                return false;
        return true;
    }
    else
        return false;
}

template<class Type>
Matrix<Type> Matrix<Type>::transpose() const
{
    Matrix<Type> result(this->theCols, this->theRows);
    for (int i = 0; i < result.theRows; i++)
    {
        for (int j = 0; j < result.theCols; j++)
            result(i, j) = (*this)(j, i);
    }

    return result;
}

template<class Type>
Matrix<Type> Matrix<Type>::element_pow(double scalar) const
{
    Matrix<Type> result(this->theRows, this->theCols);
    for (int i = 0; i < this->size(); i++)
        result.data[i] = pow(static_cast<double>(this->data[i]), scalar);
    
    return result;
}

template<class Type>
int Matrix<Type>::getRank() const
{
    Matrix<double> src = this->cast<double>();
    int main_r = 0, main_c = 0;  // position of the main element
    while (main_c < src.cols())
    {
        bool allZerosCol = false;
        if (src(main_r, main_c) == 0)
        {
            allZerosCol = true;
            for (int i = main_r + 1; i < src.rows(); i++)
            {
                if (src(i, main_c) != 0)
                {
                    src.swap_row(main_r, i);
                    allZerosCol = false;
                    break;
                }
            }
        }
        if (!allZerosCol)
        {
            for (int i = main_r + 1; i < src.rows(); i++)
            {
                double k = src(i, main_c) / src(main_r, main_c);
                for (int j = main_c; j < src.cols(); j++)
                    src(i, j) -= src(main_r, j) * k;
            }

            ++main_r;
            ++main_c;
        }
        else
        {
            ++main_c;
        }
    }

    int rank = 0;
    for (int i = 0; i < src.rows(); i++)
    {
        bool allZerosRow = true;
        for (int j = 0; j < src.cols(); j++)
        {
            if (src(i, j) != 0)
            {
                allZerosRow = false;
                break;
            }
        }
        if (!allZerosRow)
            ++rank;
    }

    return rank;
}

template<class Type>
Matrix<Type> Matrix<Type>::cofactor(int r, int c) const
{
    Matrix<Type> result(this->theRows - 1, this->theCols - 1);
    int ind = 0;
    for (int i = 0; i < this->theRows; i++)
    {
        for (int j = 0; j < this->theCols; j++)
        {
            if (r != i && c != j)
                result.data[ind++] = (*this)(i, j);
        }
    }
    
    return result;
}

template<class Type>
Matrix<Type> Matrix<Type>::inv() const
{
    if (this->theRows <= 0 || this->theCols <= 0)
        throw matrixSizeMismatch("Matrix should not be empty!");
    else if (this->theRows != this->theCols)
        throw matrixSizeMismatch("The number of rows should equal to that of cols!");
    else
    {
        Matrix<Type> result(this->theRows, this->theCols);
        double a = this->det();
        if (a == 0)
            throw matrixIsSingular();
        else
        {
            for (int i = 0; i < result.theRows; i++)
                for (int j = 0; j < result.theCols; j++)
                    result(i, j) = static_cast<Type>(std::pow(-1, i + j) * 
                                    this->cofactor(j, i).det() / a);
            
            return result;
        }
    }
}

template<class Type>
double Matrix<Type>::det() const
{
    if (this->theRows <= 0 || this->theCols <= 0)
        throw matrixSizeMismatch("Matrix should not be empty!");
    else if (this->theRows != this->theCols)
        throw matrixSizeMismatch("The number of rows should equal to that of cols!");
    else
    {
        if (this->theRows == 1)
            return (double)(this->data[0]);
        else
        {
            double sum = 0;
            for (int i = 0; i < this->theCols; i++)
            {
                sum += std::pow(-1, i) * (double)(*this)(0, i) * 
                       this->cofactor(0, i).det();
            }

            return sum;
        }
    }
}

template<class Type>
Matrix<Type> Matrix<Type>::normalize(int axis) const
{
    if (axis != 0 && axis != 1 && axis != 2)
        throw illegalParameterValue();
    else
    {
        Matrix<Type> result(theRows, theCols);
        switch (axis)
        {
        case 0:
            {
                for (int r = 0; r < theRows; r++)
                {
                    double sum = 0;
                    for (int c = 0; c < theCols; c++)
                        sum += double(this->data[r * theCols + c]);
                    for (int c = 0; c < theCols; c++)
                        result(r, c) = (Type)(double(this->data[r * theCols + c]) / sum);
                }

                break;
            }
        case 1:
            {
                for (int c = 0; c < theCols; c++)
                {
                    double sum = 0;
                    for (int r = 0; r < theRows; r++)
                        sum += double(this->data[r * theCols + c]);
                    for (int r = 0; r < theRows; r++)
                        result(r, c) = (Type)(double(this->data[r * theCols + c]) / sum);
                }

                break;
            }
        default:
            {
                double sum = this->sum();
                for (int r = 0; r < theRows; r++)
                {
                    for (int c = 0; c < theCols; c++)
                        result(r, c) = (Type)(double(this->data[r * theCols + c]) / sum);
                }

                break;
            }
        }

        return result;
    }
}

#endif //_MATRIX_HPP_
