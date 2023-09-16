/*
 * filename: Convolution.cpp
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    Definition of convolution layer's methods.
 */

#include <layers/Convolution.hpp>

namespace snn
{
    Convolution::Convolution():Layer()
    {
        input_img_rows = input_img_cols = input_img_channels = 0;
        output_img_rows = output_img_cols = output_img_channels = 0;
        filter_rows = filter_cols = 0;
        row_pads = col_pads = 0;
        row_strides = col_strides = 0;
    }

    Convolution::Convolution(int _input_img_rows, int _input_img_cols, 
                        int _input_img_channels, 
                        int _output_img_channels, 
                        int _filter_rows, int _filter_cols, 
                        int _row_pads, int _col_pads, 
                        int _row_strides, int _col_strides, 
                        double _filter_param_mean, 
                        double _filter_param_sigma, 
                        double _bias_param_mean, 
                        double _bias_param_sigma):
    Layer(_input_img_rows * _input_img_cols,
        _input_img_channels, 
        ((_input_img_rows+2*_row_pads-_filter_rows)/_row_strides+1) * 
        ((_input_img_cols+2*_col_pads-_filter_cols)/_col_strides+1), 
        _output_img_channels)
    {
        input_img_rows = _input_img_rows;
        input_img_cols = _input_img_cols;
        input_img_channels = _input_img_channels;
        output_img_rows = (_input_img_rows+2*_row_pads-_filter_rows)/_row_strides+1;
        output_img_cols = (_input_img_cols+2*_col_pads-_filter_cols)/_col_strides+1;
        output_img_channels = _output_img_channels;
        filter_rows = _filter_rows;
        filter_cols = _filter_cols;
        row_pads = _row_pads;
        col_pads = _col_pads;
        row_strides = _row_strides;
        col_strides = _col_strides;
        filter = nrandom(filter_rows * filter_cols * input_img_channels, 
                        output_img_channels, 
                        _filter_param_mean, _filter_param_sigma);
        //bias = nrandom(1, output_img_channels, 
                    //_bias_param_mean, _bias_param_sigma);
        bias = zeros<double>(1, output_img_channels);
        dFilter = zeros_like<double>(filter);
        dBias = zeros_like<double>(bias);
    }

    void Convolution::randu_filter_init(double bound1, double bound2)
    {
        filter = urandom(filter_rows, filter_cols, bound1, bound2);
    }

    void Convolution::randn_filter_init(double mean, double sigma)
    {
        filter = nrandom(filter_rows, filter_cols, mean, sigma);
    }

    void Convolution::randu_bias_init(double bound1, double bound2)
    {
        bias = urandom(1, output_img_channels, bound1, bound2);
    }

    void Convolution::randn_bias_init(double mean, double sigma)
    {
        bias = nrandom(1, output_img_channels, mean, sigma);
    }

    void Convolution::get_params_ptr(std::vector<Matrix_d*>& _params, 
                                    std::vector<Matrix_d*>& _grads)
    {
        _params.push_back(&filter);
        _params.push_back(&bias);
        _grads.push_back(&dFilter);
        _grads.push_back(&dBias);
    }

    bool Convolution::set_properties(const LayerParams& _layer_params)
    {
        if (_layer_params.size() <= 0)
            return true;

        for (LayerParams::const_iterator iter = _layer_params.begin(); 
            iter != _layer_params.end(); iter++)
        {
            auto key = iter->first;
            auto value = iter->second;

            if (key == "param_input_img_rows")
                input_img_rows = int(value);
            else if (key == "param_input_img_cols")
                input_img_cols = int(value);
            else if (key == "param_input_img_channels")
                input_img_channels = int(value);
            else if (key == "param_output_img_channels")
                output_img_channels = int(value);
            else if (key == "param_filter_rows")
                filter_rows = int(value);
            else if (key == "param_filter_cols")
                filter_cols = int(value);
            else if (key == "param_row_pads")
                row_pads = int(value);
            else if (key == "param_col_pads")
                col_pads = int(value);
            else if (key == "param_row_strides")
                row_strides = int(value);
            else if (key == "param_col_strides")
                col_strides = int(value);
            else
            {
                std::cerr << "Invalid layer property name:" << key;
                std::cerr << std::endl;
                return false;
            }
        }

        output_img_rows = (input_img_rows+2*row_pads-filter_rows)/row_strides+1;
        output_img_cols = (input_img_cols+2*col_pads-filter_cols)/col_strides+1;
        input_rows = input_img_rows * input_img_cols;
        input_cols = input_img_channels;
        output_rows = output_img_rows * output_img_cols;
        output_cols = output_img_channels;
        filter = nrandom(filter_rows * filter_cols * input_img_channels, 
                        output_img_channels, 
                        0, 0.1);
        bias = zeros<double>(1, output_img_channels);
        dFilter = zeros_like<double>(filter);
        dBias = zeros_like<double>(bias);

        return true;
    }

    bool Convolution::get_properties(LayerParams& _layer_params, 
                                    std::string& _layer_type_got)
    {
        _layer_params.clear();
        _layer_type_got = "Convolution";

        if (!_layer_params.insert(LayerParams::value_type("param_input_img_rows", 
                                                        (double)input_img_rows)).second)
            return false;
        if (!_layer_params.insert(LayerParams::value_type("param_input_img_cols", 
                                                        (double)input_img_cols)).second)
            return false;
        if (!_layer_params.insert(LayerParams::value_type("param_input_img_channels", 
                                                        (double)input_img_channels)).second)
            return false;
        if (!_layer_params.insert(LayerParams::value_type("param_output_img_channels", 
                                                        (double)output_img_channels)).second)
            return false;
        if (!_layer_params.insert(LayerParams::value_type("param_filter_rows", 
                                                        (double)filter_rows)).second)
            return false;
        if (!_layer_params.insert(LayerParams::value_type("param_filter_cols", 
                                                        (double)filter_cols)).second)
            return false;
        if (!_layer_params.insert(LayerParams::value_type("param_row_pads", 
                                                        (double)row_pads)).second)
            return false;
        if (!_layer_params.insert(LayerParams::value_type("param_col_pads", 
                                                        (double)col_pads)).second)
            return false;
        if (!_layer_params.insert(LayerParams::value_type("param_row_strides", 
                                                        (double)row_strides)).second)
            return false;
        if (!_layer_params.insert(LayerParams::value_type("param_col_strides", 
                                                        (double)col_strides)).second)
            return false;

        return true;
    }

    void Convolution::forward()
    {
        if (input_rows == 0 || input_cols == 0)
            return;
        else
        {
            // im2col
            input2col = im2col(input_data, 
                                        input_img_rows, input_img_cols, 
                                        filter_rows, filter_cols, 
                                        input_img_channels, 
                                        row_strides, col_strides, 
                                        row_pads, col_pads);
            // matrix product
            conv_output_raw = input2col * filter;
            // reshape
            conv_output = this->conv_output_transform();

            int batch_size = input_data.cols() / input_img_channels;
            output_data = zeros<double>(output_img_rows * output_img_cols, 
                                        batch_size * output_img_channels);
            for (int r = 0; r < output_data.rows(); r++)
                for (int c = 0; c < output_data.cols(); c++)
                    output_data(r, c) = conv_output(r, c) 
                                        + bias(0, c % output_img_channels);
        }
    }

    void Convolution::backward(const Matrix_d& dout)
    {
        Matrix_d dout_trans = this->dout_inverse_transform(dout);
        dFilter = input2col.transpose() * dout_trans;
        dBias = sum_axis<double>(dout_trans, 1);

        Matrix_d din2col = dout_trans * filter.transpose();
        din = col2im_add(din2col, 
                    input_img_rows, input_img_cols, 
                    filter_rows, filter_cols, 
                    input_img_channels, 
                    row_strides, col_strides, 
                    row_pads, col_pads);
    }

    Matrix_d Convolution::conv_output_transform()
    {
        int batch_size = input_data.cols() / input_img_channels;
        Matrix_d result(output_img_rows * output_img_cols, 
                        batch_size * output_img_channels);
        for (int r = 0; r < conv_output_raw.rows(); r++)
            for (int c = 0; c < conv_output_raw.cols(); c++)
                result(r % output_rows, 
                            r / output_rows * output_img_channels + c) 
                            = conv_output_raw(r, c);
            
        return result;
    }

    Matrix_d Convolution::dout_inverse_transform(const Matrix_d& dout)
    {
        Matrix_d result(conv_output_raw.rows(), conv_output_raw.cols());
        for (int r = 0; r < dout.rows(); r++)
            for (int c = 0; c < dout.cols(); c++)
                result(c / output_img_channels * output_rows + r, 
                    c % output_img_channels) = dout(r, c);
        
        return result;
    }
}

