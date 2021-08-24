/*
 * filename: Momentum.cpp
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    Definition of momentum optimizer's methods.
 */

#include <optimizers/Momentum.hpp>
#include <functions/common.hpp>

namespace snn
{
    Momentum::Momentum(std::vector<Matrix_d> _init_params, 
                    std::vector<Matrix_d> _init_grads, 
                    double _alpha, 
                    double _learning_rate):
    Optimizer(_init_params, _init_grads)
    {
        alpha = _alpha;
        learning_rate = _learning_rate;
        for (int i = 0; i < _init_params.size(); i++)
            momentums.push_back(zeros_like<double>(_init_params[i]));
    }

    void Momentum::update(std::vector<Matrix_d*>& _params, std::vector<Matrix_d*>& _grads)
    {
        for (int i = 0; i < _params.size(); i++)
        {
            momentums[i] = momentums[i] * alpha + *(_grads[i]) * learning_rate;
            *(_params[i]) = *(_params[i]) - momentums[i];
        }
    }
}

