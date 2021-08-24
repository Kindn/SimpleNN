/*
 * filename: SoftmaxWithLoss.cpp
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    Definition of cross-entropy-loss layer's methods.
 */

#include <loss/SoftmaxWithLoss.hpp>

namespace snn
{
    void SoftmaxWithLoss::forward()
    {
        softmax_output = softmax(input_data, 1);

        if (is_one_hot_label())
        {
            double sum = 0;
            for (int i = 0; i < one_hot_labels.cols(); i++)
            {
                int one_hot_index = one_hot_labels(0, i);
                sum += (-log(softmax_output(one_hot_index, i)));
            }
            loss = sum / double(input_data.cols());
        }
        else
        {
            loss = crossEntropyLoss(softmax_output, 
                                        normalized_labels) 
                        / double(input_data.cols());
        }

    }

    void SoftmaxWithLoss::backward()
    {
        din = zeros<double>(input_data.rows(), input_data.cols());
        if (is_one_hot_label())
        {
            for (int c = 0; c < din.cols(); c++)
            {
                int one_hot_index = one_hot_labels(0, c);
                for (int r = 0; r < din.rows(); r++)
                {
                    double y = softmax_output(r, c);
                    double t = (r == one_hot_index) ? 1.0 : 0;

                    din(r, c) = (y - t) / double(input_data.cols());
                }
            }
        }
        else
        {
            for (int c = 0; c < din.cols(); c++)
            {
                for (int r = 0; r < din.rows(); r++)
                {
                    double y = softmax_output(r, c);
                    double t = normalized_labels(r, c);

                    din(r, c) = (y - t) / double(input_data.cols());
                }
            }
        }
    }
}

