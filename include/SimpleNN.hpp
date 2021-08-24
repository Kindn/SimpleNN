/*
 * filename: SimpleNN.hpp
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    
 */

#ifndef  _SIMPLE_NN_HPP_
#define  _SIMPLE_NN_HPP_

#include <Matrix/Matrix.hpp>
#include <Matrix/exceptions.hpp>

#include <functions/functions.hpp>
#include <functions/common.hpp>
#include <functions/io.hpp>

#include <layers/Layer.hpp>
#include <layers/Affine.hpp>
#include <layers/Convolution.hpp>
#include <layers/LRelu.hpp>
#include <layers/MaxPooling.hpp>
#include <layers/Relu.hpp>
#include <layers/Sigmoid.hpp>
#include <layers/Softmax.hpp>

#include <loss/Loss.hpp>
#include <loss/CrossEntropyLoss.hpp>
#include <loss/SoftmaxWithLoss.hpp>

#include <optimizers/Optimizer.hpp>
#include <optimizers/GradDescent.hpp>
#include <optimizers/Momentum.hpp>
#include <optimizers/AdaGrad.hpp>

#include <nets/Dataset.hpp>
#include <nets/Label.hpp>
#include <nets/Sequential.hpp>


#endif //_SIMPLE_NN_HPP_
