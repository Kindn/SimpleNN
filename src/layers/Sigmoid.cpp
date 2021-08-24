/*
 * filename: Sigmoid.cpp
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    Definition of sigmoid layer's methods.
 */

#include <layers/Sigmoid.hpp>

namespace snn
{
    Sigmoid::Sigmoid():Layer()
    {

    }

    Sigmoid::Sigmoid(int _data_rows, int _data_cols):
    Layer(_data_rows, _data_cols, _data_rows, _data_cols)
    {

    }

    void Sigmoid::get_params_ptr(std::vector<Matrix_d*>& _params, 
                                        std::vector<Matrix_d*>& _grads)
    {
        
    }

    bool Sigmoid::set_properties(LayerParams& _layer_params)
    {
        if (_layer_params.size() <= 0)
            return true;
        else
        {
            std::cerr << "Invalid layer property name:" << _layer_params.begin()->first;
            std::cerr << std::endl;
            return false;
        }
    }

    bool Sigmoid::get_properties(LayerParams& _layer_params, 
                                std::string& _layer_type_got)
    {
        _layer_params.clear();
        _layer_type_got = "Sigmoid";

        return true;
    }

    void Sigmoid::forward()
    {
        output_data = sigmoid(input_data);
    }

    void Sigmoid::backward(Matrix_d& dout)
    {
        din = *(new Matrix_d(input_data.rows(), input_data.cols()));
        for (int r = 0; r < din.rows(); r++)
        {
            for (int c = 0; c < din.cols(); c++)
                din(r, c) = output_data(r, c) * (1.0 - output_data(r, c))
                            * dout(r, c);
        }
    }
}


