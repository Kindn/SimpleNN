/*
 * filename: Convolution.cpp
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    Definition of max-pooling layer's methods.
 */

#include <layers/MaxPooling.hpp>

namespace snn
{
    MaxPooling::MaxPooling():Layer()
    {
        input_img_rows = input_img_cols = 0;
        output_img_rows = output_img_cols = 0;
        filter_rows = filter_cols = 0;
        row_pads = col_pads = 0;
        row_strides = col_strides = 0;
        output_flatten_flag = false;
    }

    MaxPooling::MaxPooling(int _input_img_rows, int _input_img_cols, 
                        int _filter_rows, int _filter_cols,
                        int _row_pads, int _col_pads, 
                        int _row_strides, int _col_strides, 
                        int _image_channels, 
                        bool _output_flatten_flag):
    Layer(_input_img_rows * _input_img_cols,
        _image_channels, 
        ((_input_img_rows+2*_row_pads-_filter_rows)/_row_strides+1) * 
        ((_input_img_cols+2*_col_pads-_filter_cols)/_col_strides+1), 
        _image_channels)
    {
        input_img_rows = _input_img_rows;
        input_img_cols = _input_img_cols;
        output_img_rows = (_input_img_rows+2*_row_pads-_filter_rows)/_row_strides+1;
        output_img_cols = (_input_img_cols+2*_col_pads-_filter_cols)/_col_strides+1;
        filter_rows = _filter_rows;
        filter_cols = _filter_cols;
        row_pads = _row_pads;
        col_pads = _col_pads;
        row_strides = _row_strides;
        col_strides = _col_strides;
        image_channels = _image_channels;
        output_flatten_flag = _output_flatten_flag;
    }

    void MaxPooling::get_params_ptr(std::vector<Matrix_d*>& _params, 
                                        std::vector<Matrix_d*>& _grads)
    {
        
    }

    bool MaxPooling::set_properties(const LayerParams& _layer_params)
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
            else if (key == "param_image_channels")
                image_channels = int(value);
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
            else if (key == "param_output_flatten")
                output_flatten_flag = bool(value);
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
        input_cols = image_channels;
        output_rows = output_img_rows * output_img_cols;
        output_cols = image_channels;

        return true;
    }

    bool MaxPooling::get_properties(LayerParams& _layer_params, 
                                    std::string& _layer_type_got)
    {
        _layer_params.clear();
        _layer_type_got = "MaxPooling";

        if (!_layer_params.insert(LayerParams::value_type("param_input_img_rows", 
                                                        (double)input_img_rows)).second)
            return false;
        if (!_layer_params.insert(LayerParams::value_type("param_input_img_cols", 
                                                        (double)input_img_cols)).second)
            return false;
        if (!_layer_params.insert(LayerParams::value_type("param_image_channels", 
                                                        (double)image_channels)).second)
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
        if (!_layer_params.insert(LayerParams::value_type("param_output_flatten", 
                                                        (double)output_flatten_flag)).second)
            return false;
        
        return true;
    }

    void MaxPooling::forward()
    {
        if (input_rows == 0 || input_cols == 0)
            return;
        else
        {
            input2col = im2col(input_data, 
                            input_img_rows, input_img_cols, 
                            filter_rows, filter_cols, 
                            1, 
                            row_strides, col_strides, 
                            row_pads, col_pads);
            output_data = argmax_axis<double>(input2col, 
                                            max_cols, 
                                            0).transpose().reshape(input_data.cols(), 
                                                                    output_rows).transpose();
            if (output_flatten_flag)
                output_data = output_data.transpose().reshape(input_data.cols() / image_channels, 
                                                            output_rows * image_channels).transpose();
        }
    }

    void MaxPooling::backward(const Matrix_d& dout)
    {
        Matrix_d actual_dout = dout;
        if (output_flatten_flag)
            actual_dout = dout.transpose().reshape(dout.cols() * image_channels, 
                                            dout.rows() / image_channels).transpose();
        Matrix_d din_i2c(input2col.rows(), input2col.cols());
        for (int i = 0; i < actual_dout.rows(); i++)
        {
            for (int j = 0; j < actual_dout.cols(); j++)
            {
                int row_in_i2c = i + actual_dout.rows() * j;
                int col_in_i2c = max_cols[row_in_i2c];
                din_i2c(row_in_i2c, col_in_i2c) = actual_dout(i, j);
            }
        }
        din = col2im_add(din_i2c, 
                        input_img_rows, input_img_cols, 
                        filter_rows, filter_cols, 
                        1, 
                        row_strides, col_strides, 
                        row_pads, col_pads);
    }
}

