/*
 * filename: Layer.hpp
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    Declaration of basic class Optimizer.
 */

#ifndef  _OPTIMIZER_HPP_
#define  _OPTIMIZER_HPP_

#include <Matrix/Matrix.hpp>
#include <functions/functions.hpp>
#include <vector>
#include <memory>

namespace snn
{
    class Optimizer : public std::enable_shared_from_this<Optimizer>
    {
        private:
            std::vector<Matrix_d> init_params;
            std::vector<Matrix_d> init_grads;

        public:
            Optimizer(std::vector<Matrix_d> _init_params, 
                            std::vector<Matrix_d> init_grads);
            virtual ~Optimizer() {}

            void set_optimizer(std::vector<Matrix_d> _init_params, 
                            std::vector<Matrix_d> init_grads);

            virtual void update(std::vector<Matrix_d*>& params, 
                                std::vector<Matrix_d*>& grads) = 0;

    };
}



#endif //_OPTIMIZER_HPP_

