/*
 * filename: AdaGrad.cpp
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    Definition of AdaGrad optimizer's methods.
 */

#include <optimizers/AdaGrad.hpp>

namespace snn
{
    AdaGrad::AdaGrad(std::vector<Matrix_d> _init_params, 
                     std::vector<Matrix_d> _init_grads, 
                     double _learning_rate):
    Optimizer(_init_params, _init_grads)
    {
        learning_rate = _learning_rate;
        h.clear();
        for (int i = 0; i < _init_grads.size(); i++)
            h.push_back(_init_grads[i].element_pow(2.0));
    }

    void AdaGrad::update(std::vector<Matrix_d*>& _params, std::vector<Matrix_d*>& _grads)
    {
        const double eps = 1e-7;
        for (int i = 0; i < _params.size(); i++)
        {
            h[i] += (*(_grads[i])).element_pow(2.0);
            Matrix_d den = (h[i] + eps).element_pow(0.5);
            *(_params[i]) -= (*(_grads[i]) / den) * learning_rate;
        }
    }
}


