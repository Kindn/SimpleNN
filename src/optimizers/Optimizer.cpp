/*
 * filename: Optimizer.hpp
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    Definition of basic optimizer's methods.
 */

#include <optimizers/Optimizer.hpp>

namespace snn
{
    Optimizer::Optimizer(std::vector<Matrix_d> _init_params, 
                        std::vector<Matrix_d> _init_grads)
    {
        this->init_params = _init_params;
        this->init_grads = _init_grads;
    }

    void Optimizer::set_optimizer(std::vector<Matrix_d> _init_params, 
                    std::vector<Matrix_d> _init_grads)
    {
        this->init_params = _init_params;
        this->init_grads = _init_grads;
    }
}


