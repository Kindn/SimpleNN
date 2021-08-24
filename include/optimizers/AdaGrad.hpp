/*
 * filename: AdaGrad.hpp
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    Declaration of AdaGrad optimizer.
 */

#ifndef  _ADAGRAD_HPP_
#define  _ADAGRAD_HPP_

#include <optimizers/Optimizer.hpp>
#include <Matrix/Matrix.hpp>
#include <functions/functions.hpp>
#include <vector>

namespace snn
{
    class AdaGrad : public Optimizer
    {
        public:
            double learning_rate;
            std::vector<Matrix_d> h;

        public:
            AdaGrad(std::vector<Matrix_d> _init_params, 
                        std::vector<Matrix_d> _init_grads, 
                        double _learning_rate);

            virtual void update(std::vector<Matrix_d*>& params, 
                                std::vector<Matrix_d*>& grads) override;
    };
}



#endif //_ADAGRAD_HPP_
