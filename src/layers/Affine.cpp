/*
 * filename: Affine.hpp
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    Definition of affine layer's methods.
 */

#include <layers/Affine.hpp>

namespace snn
{
    Affine::Affine():Layer()
    {
        w_rows = output_rows;
        w_cols = input_rows;
        b_rows = output_rows;
        b_cols = input_cols;
    }

    Affine::Affine(int _input_rows, int _input_cols,
                int _output_rows, int _output_cols, 
                double _mean_w, double _sigma_w, 
                double _mean_b, double _sigma_b):
    Layer(_input_rows, _input_cols, _output_rows, _output_cols)
    {
        w_rows = output_rows;
        w_cols = input_rows;
        b_rows = output_rows;
        b_cols = output_cols;
        
        weights = nrandom(w_rows, w_cols, _mean_w, _sigma_w);
        //bias = nrandom(b_rows, b_cols, _mean_b, _sigma_b);
        bias = zeros<double>(b_rows, b_cols);
        dWeights = zeros_like<double>(weights);
        dBias = zeros_like<double>(bias);
    }

    void Affine::randu_weights_init(double bound1, double bound2)
    {
        weights = urandom(w_rows, w_cols, bound1, bound2);
    }

    void Affine::randn_weights_init(double mean, double sigma)
    {
        weights = nrandom(w_rows, w_cols, mean, sigma);
    }

    void Affine::randu_bias_init(double bound1, double bound2)
    {
        bias = urandom(b_rows, b_cols, bound1, bound2);
    }

    void Affine::randn_bias_init(double mean, double sigma)
    {
        bias = nrandom(b_rows, b_cols, mean, sigma);
    }

    void Affine::get_params_ptr(std::vector<Matrix_d*>& _params, 
                                std::vector<Matrix_d*>& _grads)
    {
        _params.push_back(&weights);
        _params.push_back(&bias);
        _grads.push_back(&dWeights);
        _grads.push_back(&dBias);
    }

    bool Affine::set_properties(LayerParams& _layer_params)
    {
        if (_layer_params.size() <= 0)
            return true;

        for (LayerParams::iterator iter = _layer_params.begin(); 
            iter != _layer_params.end(); iter++)
        {
            auto key = iter->first;
            auto value = iter->second;

            if (key == "param_input_rows")
                input_rows = int(value);
            else if (key == "param_input_cols")
                input_cols = int(value);
            else if (key == "param_output_rows")
                output_rows = int(value);
            else if (key == "param_output_cols")
                output_cols = int(value);
            else
            {
                std::cerr << "Invalid layer property name:" << key;
                std::cerr << std::endl;
                return false;
            }
        }
        
        w_rows = output_rows;
        w_cols = input_rows;
        b_rows = output_rows;
        b_cols = output_cols;
        weights = nrandom(w_rows, w_cols, 0, 0.01);
        bias = zeros<double>(b_rows, b_cols);
        dWeights = zeros_like<double>(weights);
        dBias = zeros_like<double>(bias);

        return true;
    }

    bool Affine::get_properties(LayerParams& _layer_params, 
                                std::string& _layer_type_got)
    {
        _layer_params.clear();
        _layer_type_got = "Affine";

        if (!_layer_params.insert(LayerParams::value_type("param_input_rows", 
                                                        (double)input_rows)).second)
            return false;
        if (!_layer_params.insert(LayerParams::value_type("param_input_cols", 
                                                        (double)input_cols)).second)
            return false;
        if (!_layer_params.insert(LayerParams::value_type("param_output_rows", 
                                                        (double)output_rows)).second)
            return false;
        if (!_layer_params.insert(LayerParams::value_type("param_output_cols", 
                                                        (double)output_cols)).second)
            return false;

        return true;
    }

    void Affine::forward()
    {
        Matrix_d w_sum = weights * input_data;
        output_data = zeros_like<double>(w_sum);
        for (int r = 0; r < w_sum.rows(); r++)
        {
            for (int c = 0; c < w_sum.cols(); c++)
            {
                output_data(r, c) = w_sum(r, c) + bias(r, 0);
            }
        }
    }

    void Affine::backward(Matrix_d& dout)
    {
        din = weights.transpose() * dout;
        dWeights = dout * input_data.transpose();
        dBias = sum_axis<double>(dout, 0);
    }
}


