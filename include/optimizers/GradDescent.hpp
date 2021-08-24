/*
 * filename: GradDescent.hpp
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    Declaration of Stochastic Gradient Decent optimizer.
 */

#ifndef  _GRAD_DESCENT_HPP_
#define _GRAD_DESCENT_HPP_

#include <optimizers/Optimizer.hpp>
#include <Matrix/Matrix.hpp>
#include <functions/functions.hpp>
#include <vector>

namespace snn
{
    class GradDescent : public Optimizer
    {
        public:
            double learning_rate;

        public:
            GradDescent(std::vector<Matrix_d> _init_params, 
                        std::vector<Matrix_d> _init_grads, 
                        double _learning_rate);

            virtual void update(std::vector<Matrix_d*>& params, 
                                std::vector<Matrix_d*>& grads) override;
    };
}




#endif //_GRAD_DESCENT_HPP_
