/*
 * filename: SoftmaxWithLoss.hpp
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    Declaration of softmax-with-loss layer.
 */

#ifndef  _SOFTMAX_WITH_LOSS_HPP_
#define  _SOFTMAX_WITH_LOSS_HPP_

#include <loss/Loss.hpp>
#include <Matrix/Matrix.hpp>
#include <functions/functions.hpp>
#include <vector>

namespace snn
{
    class SoftmaxWithLoss : public Loss
    {
        private:
            Matrix_d softmax_output;
            
        public:
            SoftmaxWithLoss(bool _one_hot_flag):Loss(_one_hot_flag)
            {}

            virtual void forward() override;
            virtual void backward() override;
    };

}



#endif //_SOFTMAX_WITH_LOSS_HPP_
