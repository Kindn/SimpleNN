/*
 * filename: Convolution.hpp
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    Declaration of convolution layer.
 */

#ifndef  _CONVOLUTION_HPP_
#define  _CONVOLUTION_HPP_

#include <layers/Layer.hpp>
#include <Matrix/Matrix.hpp>
#include <functions/functions.hpp>
#include <functions/common.hpp>
#include <vector>

namespace snn
{
    class Convolution : public Layer
    {
        public:
            int input_img_rows, input_img_cols;
            int input_img_channels;
            int output_img_rows, output_img_cols;
            int output_img_channels;

            int filter_rows, filter_cols;
            Matrix_d filter, dFilter;
            Matrix_d bias, dBias;
            int row_pads, col_pads;
            int row_strides, col_strides;

            
        public:
            Convolution();
            Convolution(int _input_img_rows, int _input_img_cols, 
                        int _input_img_channels, 
                        int _output_img_channels, 
                        int _filter_rows, int _filter_cols, 
                        int _row_pads, int _col_pads, 
                        int _row_strides, int _col_strides, 
                        double _filter_param_mean = 0, 
                        double _filter_param_sigma = 1.0, 
                        double _bias_param_mean = 0, 
                        double _bias_param_sigma = 1.0);
            ~Convolution() {};

            void randu_filter_init(double bound1, double bound2);
            void randn_filter_init(double mean, double sigma);
            void randu_bias_init(double bound1, double bound2);
            void randn_bias_init(double mean, double sigma);

            virtual void get_params_ptr(std::vector<Matrix_d*>& _params, 
                                        std::vector<Matrix_d*>& _grads) override;

            virtual bool set_properties(const LayerParams& _layer_params)  override;
            virtual bool get_properties(LayerParams& _layer_params, 
                                        std::string& _layer_type_got)  override;
            virtual void forward()  override;
            virtual void backward(const Matrix_d& dout)  override;

        private:
            Matrix_d input2col;
            Matrix_d conv_output_raw;
            Matrix_d conv_output;


            Matrix_d conv_output_transform();
            Matrix_d dout_inverse_transform(const Matrix_d& dout);
    };
}



#endif //_CONVOLUTION_HPP_
