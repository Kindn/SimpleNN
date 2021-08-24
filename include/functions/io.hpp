/*
 * filename: io.hpp
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    Functions about interaction between data and files.
 */

#ifndef  _IO_HPP_
#define  _IO_HPP_

#include <Matrix/Matrix.hpp>
#include <data/csv.hpp>
#include <vector>
#include <string>
#include <fstream>

namespace snn
{
    template<class DType>
    bool csv2mat(const std::string& _file_path, 
                const std::string& _delimeter, 
                Matrix<DType>& _output_mat);


    /***********************************************************************/

    template<class DType>
    bool csv2mat(const std::string& _file_path, 
                const std::string& _delimeter, 
                Matrix<DType>& _output_mat)
    {
        std::vector<std::vector<DType>> vec_data;
        std::ifstream input_file(_file_path);
        if (input_file.fail())
        {
            std::cout << "In csv2mat: File not found." << std::endl;
            return false;
        }
        std::string row_str;
        int cols;
        
        while (std::getline(input_file, row_str))
        {
            std::vector<std::string> seg_line;
            csv_seg_line(row_str, seg_line);
            
            if (seg_line.size())
            {
                std::vector<DType> row_vec;
                for (int i = 0; i < seg_line.size(); i++)
                {
                    DType value = static_cast<DType>(atof(seg_line[i].c_str()));
                    row_vec.push_back(value);
                }
                vec_data.push_back(row_vec);
            }
            else
            {
                return false;
            }

            cols = vec_data[0].size();
            if (seg_line.size() != cols)
                return false;
        }

        int rows = vec_data.size();
        Matrix<DType> result(rows, cols);
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                result(r, c) = vec_data[r][c];
        
        _output_mat = result;

        return true;
        
    }
}

#endif //_IO_HPP_
