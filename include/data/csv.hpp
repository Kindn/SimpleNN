/*
 * filename: csv.hpp
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    Functions about csv procession.
 */

#ifndef  _CSV_HPP_
#define  _CSV_HPP_

#include <vector>
#include <string>
#include <fstream>

void csv_seg_line(const std::string& _line_str, 
                  std::vector<std::string>& _output_vec);

#endif //_CSV_HPP_
