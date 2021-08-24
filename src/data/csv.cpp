/*
 * filename: csv.cpp
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    Definition of functions about csv procession.
 */

#include <data/csv.hpp>

void csv_seg_line(std::string& _line_str, 
                  std::vector<std::string>& _output_vec)
{
    int line_size = _line_str.length();
    int curr_pos = 0;
    std::string element = "";
    _output_vec.clear();
    char curr_char;
    while (1)
    {
        curr_char = _line_str[curr_pos];
        if (isdigit(curr_char) || curr_char == '.' ||
            curr_char == '+' || curr_char == '-')
            element += curr_char;
        else if (element != "")
        {
            _output_vec.push_back(element);
            element = "";
        }

        curr_pos++;
        if (curr_pos >= line_size)
        {
            if (element != "") _output_vec.push_back(element);
            break;
        }
            
    }
    
}
