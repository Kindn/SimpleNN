/*
 * filename: LRelu.hpp
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    Declaration of leaky-relu layer.
 */

#ifndef  _LRELU_HPP_
#define  _LRELU_HPP_

#include <layers/Layer.hpp>
#include <Matrix/Matrix.hpp>
#include <functions/functions.hpp>
#include <functions/common.hpp>

namespace snn
{
    class LRelu : public Layer
    {
        public:
            double a;

        public:
            LRelu();
            LRelu(int _data_rows, int _data_cols, double a = 0.1);

            virtual void get_params_ptr(std::vector<Matrix_d*>& _params, 
                                        std::vector<Matrix_d*>& _grads) override;

            virtual bool set_properties(const LayerParams& _layer_params)  override;
            virtual bool get_properties(LayerParams& _layer_params, 
                                        std::string& _layer_type_got)  override;
            virtual void forward()  override;
            virtual void backward(const Matrix_d& dout)  override;
            
    };

}

#endif //_LRELU_HPP_
