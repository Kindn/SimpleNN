/*
 * filename: MaxPooing.hpp
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    Declaration of max-pooling layer.
 */

#ifndef  _MAX_POOLING_HPP_
#define  _MAX_POOLING_HPP_

#include <layers/Layer.hpp>
#include <Matrix/Matrix.hpp>
#include <functions/functions.hpp>
#include <functions/common.hpp>
#include <vector>

namespace snn
{
    class MaxPooling : public Layer
    {
        public:
            int input_img_rows, input_img_cols;
            int output_img_rows, output_img_cols;
            int filter_rows, filter_cols;
            int row_pads, col_pads;
            int row_strides, col_strides;
            int image_channels;

            bool output_flatten_flag;


        public:
            MaxPooling();
            MaxPooling(int _input_img_rows, int _input_img_cols, 
                        int _filter_rows, int _filter_cols, 
                        int _row_pads, int _col_pads, 
                        int _row_strides, int _col_strides, 
                        int _image_channels, 
                        bool _output_flatten_flag = false);

            bool is_output_flatten() const
            {return this->output_flatten_flag;}

            virtual void get_params_ptr(std::vector<Matrix_d*>& _params, 
                                        std::vector<Matrix_d*>& _grads) override;

            virtual bool set_properties(LayerParams& _layer_params)  override;
            virtual bool get_properties(LayerParams& _layer_params, 
                                        std::string& _layer_type_got)  override;
            virtual void forward()  override;
            virtual void backward(Matrix_d& dout)  override;

        private:
            Matrix_d input2col;
            std::vector<int> max_cols;
            
    };
}



#endif //_MAX_POOLING_HPP_
