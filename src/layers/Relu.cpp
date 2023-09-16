/*
 * filename: Relu.hpp
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    Definition of relu layer's methods.
 */

#include <layers/Relu.hpp>

namespace snn
{
    Relu::Relu():Layer()
    {

    }

    Relu::Relu(int _data_rows, int _data_cols):
    Layer(_data_rows, _data_cols, _data_rows, _data_cols)
    {

    }

    void Relu::get_params_ptr(std::vector<Matrix_d*>& _params, 
                            std::vector<Matrix_d*>& _grads)
    {
        
    }

    bool Relu::set_properties(const LayerParams& _layer_params)
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

    bool Relu::get_properties(LayerParams& _layer_params, 
                            std::string& _layer_type_got)
    {
        _layer_params.clear();
        _layer_type_got = "Relu";

        return true;
    }

    void Relu::forward()
    {
        output_data = relu(input_data);
    }

    void Relu::backward(const Matrix_d& dout)
    {
        din = zeros_like<double>(dout);
        for (int r = 0; r < din.rows(); r++)
            for (int c = 0; c < din.cols(); c++)
                din(r, c) = ((input_data(r, c) > 0) ? 1.0 : 0) * dout(r, c);
    }
}

