/*
 * filename: Softmax.hpp
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    Declaration of softmax Layer.
 */
#ifndef  _SOFTMAX_HPP_
#define  _SOFTMAX_HPP_

#include <layers/Layer.hpp>
#include <Matrix/Matrix.hpp>
#include <functions/functions.hpp>
#include <vector>

namespace snn
{
    class Softmax : public Layer
    {
        public:
            
        public:
            Softmax() {}

            virtual void get_params_ptr(std::vector<Matrix_d*>& _params, 
                                        std::vector<Matrix_d*>& _grads) override;
            
            virtual bool set_properties(const LayerParams& _layer_params)  override;
            virtual bool get_properties(LayerParams& _layer_params, 
                                        std::string& _layer_type_got)  override;
            virtual void forward()  override;
            virtual void backward(const Matrix_d& dout)  override;
    };
}



#endif // _SOFTMAX_HPP_
