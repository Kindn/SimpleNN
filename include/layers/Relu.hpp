/*
 * filename: Relu.hpp
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    Declaration of relu layer.
 */

#ifndef  _RELU_HPP_
#define  _RELU_HPP_

#include <layers/Layer.hpp>
#include <Matrix/Matrix.hpp>
#include <functions/functions.hpp>
#include <functions/common.hpp>

namespace snn
{
    class Relu : public Layer
    {
        public:
            Relu();
            Relu(int _data_rows, int _data_cols);

            virtual void get_params_ptr(std::vector<Matrix_d*>& _params, 
                                        std::vector<Matrix_d*>& _grads) override;

            virtual bool set_properties(LayerParams& _layer_params)  override;
            virtual bool get_properties(LayerParams& _layer_params, 
                                        std::string& _layer_type_got)  override;
            virtual void forward()  override;
            virtual void backward(Matrix_d& dout)  override;
            
    };

}

#endif //_RELU_HPP_
