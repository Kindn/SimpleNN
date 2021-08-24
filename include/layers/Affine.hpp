/*
 * filename: Affine.hpp
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    Declaration of affine layer.
 */

#ifndef  _AFFINE_HPP_
#define  _AFFINE_HPP_

#include <layers/Layer.hpp>
#include <Matrix/Matrix.hpp>
#include <functions/common.hpp>
#include <functions/functions.hpp>

namespace snn
{
    class Affine : public Layer
    {
        public:
            int w_rows, w_cols;
            int b_rows, b_cols;

            Matrix_d weights, bias;
            Matrix_d dWeights, dBias;

        public:
            Affine();
            Affine(int _input_rows, int _input_cols,
                int _output_rows, int _output_cols, 
                double mean_w = 0, double sigma_w = 0.01, 
                double mean_b = 0, double sigma_b = 0.01);

            void randu_weights_init(double bound1, double bound2);
            void randn_weights_init(double mean, double sigma);
            void randu_bias_init(double bound1, double bound2);
            void randn_bias_init(double mean, double sigma);
            
            virtual void get_params_ptr(std::vector<Matrix_d*>& _params, 
                                        std::vector<Matrix_d*>& _grads) override;

            virtual bool set_properties(LayerParams& _layer_params)  override;
            virtual bool get_properties(LayerParams& _layer_params, 
                                        std::string& _layer_type_got)  override;
            virtual void forward()  override;
            virtual void backward(Matrix_d& dout)  override;
    };

}



#endif //_AFFFINE_HPP_
