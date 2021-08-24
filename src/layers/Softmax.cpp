/*
 * filename: Softmax.hpp
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    Definition of softmax layer's methods.
 */

#include <layers/Softmax.hpp>
#include <functions/common.hpp>

namespace snn
{
    void Softmax::get_params_ptr(std::vector<Matrix_d*>& _params, 
                                    std::vector<Matrix_d*>& _grads)
    {
        return;
    }                    

    bool Softmax::set_properties(LayerParams& _layer_params)
    {
        if (_layer_params.size() <= 0)
            return true;
        else
        {
            std::cerr << "Invalid layer property name:";
            std::cerr << _layer_params.begin()->first;
            std::cerr << std::endl;
            return false;
        }
    }

    bool Softmax::get_properties(LayerParams& _layer_params, 
                                std::string& _layer_type_got)
    {
        _layer_params.clear();
        _layer_type_got = "Softmax";

        return true;
    }

    void Softmax::forward()
    {
        output_data = softmax(input_data, 1);
    }

    void Softmax::backward(Matrix_d& dout)
    {
        din = zeros<double>(input_data.rows(), input_data.cols());
        for (int c = 0; c < output_data.cols(); c++)
        {
            for (int r = 0; r < output_data.rows(); r++)
            {
                for (int k = 0; k < dout.rows(); k++)
                {
                    if (r == k)
                        din(r, c) += output_data(k, c) * (1.0 - output_data(k, c)) 
                                        * dout(k, c);
                    else
                        din(r, c) += -output_data(r, c) * output_data(k, c) 
                                        * dout(k, c);
                }
            }
        }
    }
}

