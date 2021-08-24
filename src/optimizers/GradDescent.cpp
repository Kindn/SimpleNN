/*
 * filename: GradDescent.cpp
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    Definition of Stochastic Gradient Descent optimizer's methods.
 */

#include <optimizers/GradDescent.hpp>

namespace snn
{
    GradDescent::GradDescent(std::vector<Matrix_d> _init_params, 
                             std::vector<Matrix_d> _init_grads, 
                             double _learning_rate):
    Optimizer(_init_params, _init_grads)
    {
        learning_rate = _learning_rate;
    }

    void GradDescent::update(std::vector<Matrix_d*>& _params, std::vector<Matrix_d*>& _grads)
    {
        for (int i = 0; i < _params.size(); i++)
            *(_params[i]) -= (*(_grads[i]) * this->learning_rate);
    }
}


