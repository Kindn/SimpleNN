/*
 * filename: Layer.cpp
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    Defition of some member functions of basic class Layer.
 */

#include <layers/Layer.hpp>

namespace snn
{
    Layer::Layer()
    {
        input_rows = input_cols = 0;
        output_rows = output_cols = 0;
        //din_rows = input_rows;
        //din_cols = input_cols;
    }

    Layer::Layer(int _input_rows, int _input_cols,
                int _output_rows, int _output_cols)
    {
        input_rows = _input_rows;
        input_cols = _input_cols;
        output_rows = _output_rows;
        output_cols = _output_cols;
        //din_rows = input_rows;
        //din_cols = input_cols;
    }
}

