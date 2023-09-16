/*
 * filename: Relu.hpp
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    Definition of relu layer's methods.
 */

#include <layers/LRelu.hpp>

namespace snn
{
    LRelu::LRelu():Layer()
    {
        a = 0.1;
    }

    LRelu::LRelu(int _data_rows, int _data_cols, double _a):
    Layer(_data_rows, _data_cols, _data_rows, _data_cols)
    {
        a = _a;
    }

    void LRelu::get_params_ptr(std::vector<Matrix_d*>& _params, 
                                std::vector<Matrix_d*>& _grads)
    {
        
    }

    bool LRelu::set_properties(const LayerParams& _layer_params)
    {
        if (_layer_params.size() <= 0)
            return true;

        for (LayerParams::const_iterator iter = _layer_params.begin(); 
            iter != _layer_params.end(); iter++)
        {
            auto key = iter->first;
            auto value = iter->second;

            if (key == "param")
                a = double(value);
            else
            {
                std::cerr << "Invalid layer property name:" << key;
                std::cerr << std::endl;
                return false;
            }
        }

        return true;
    }

    bool LRelu::get_properties(LayerParams& _layer_params, 
                            std::string& _layer_type_got)
    {
        _layer_params.clear();
        _layer_type_got = "LRelu";

        if (!_layer_params.insert(LayerParams::value_type("param_a", 
                                                        a)).second)
            return false;

        return true;
    }

    void LRelu::forward()
    {
        output_data = leaky_relu(input_data, a);
    }

    void LRelu::backward(const Matrix_d& dout)
    {
        din = zeros_like<double>(dout);
        for (int r = 0; r < din.rows(); r++)
            for (int c = 0; c < din.cols(); c++)
                din(r, c) = ((input_data(r, c) > 0) ? 1.0 : (1.0 / a)) * dout(r, c);
    }
}

