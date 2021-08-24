/*
 * filename: CrossEntropyLoss.hpp
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    Declaration of cross-entropy-loss layer.
 */

#ifndef  _CROSS_ENTROPY_LOSS_HPP_
#define  _CROSS_ENTROPY_LOSS_HPP_

#include <loss/Loss.hpp>
#include <Matrix/Matrix.hpp>
#include <functions/functions.hpp>
#include <vector>

namespace snn
{
    class CrossEntropyLoss : public Loss
    {
        public:
            CrossEntropyLoss(bool _one_hot_flag):Loss(_one_hot_flag)
            {}

            virtual void forward() override;
            virtual void backward() override;
    };
}




#endif //_CROSS_ENTROPY_LOSS_HPP_
