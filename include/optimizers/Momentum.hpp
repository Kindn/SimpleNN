/*
 * filename: Momentum.hpp
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    Declaration of momentum optimizer.
 */

#ifndef  _MOMENTUM_HPP_
#define  _MOMENTUM_HPP_

#include <optimizers/Optimizer.hpp>
#include <Matrix/Matrix.hpp>
#include <functions/functions.hpp>
#include <vector>

namespace snn
{
    class Momentum : public Optimizer
    {
        public:
            std::vector<Matrix_d> momentums;
            double alpha;
            double learning_rate;

        public:
            Momentum(std::vector<Matrix_d> _init_params, 
                    std::vector<Matrix_d> _init_grads, 
                    double _alpha, 
                    double _learning_rate);

            virtual void update(std::vector<Matrix_d*>& params, 
                                std::vector<Matrix_d*>& grads) override;
    };
}


#endif //_MOMENTUM_HPP_
