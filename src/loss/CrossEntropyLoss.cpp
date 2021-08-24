/*
 * filename: CrossEntropyLoss.cpp
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    Definition of cross-entropy-loss layer's methods.
 */

#include <loss/CrossEntropyLoss.hpp>

namespace snn
{
    void CrossEntropyLoss::forward()
    {
        loss = 0;
        const double eps = 1e-7;
        int input_cols = input_data.cols();
        if (is_one_hot_label())
        {
            //std::cout << one_hot_labels << std::endl;
            double sum = 0;
            for (int i = 0; i < one_hot_labels.cols(); i++)
            {
                int one_hot_index = one_hot_labels(0, i);
                sum += (-log(input_data(one_hot_index, i) + eps));
            }
            loss = sum / double(input_cols);
        }
        else
        {
            loss = crossEntropyLoss(input_data, 
                                        normalized_labels)
                        / double(input_cols);
        }
    }

    void CrossEntropyLoss::backward()
    {
        Matrix_d din_temp(input_data.rows(), input_data.cols());
        const double eps = 1e-7;
        int input_cols = input_data.cols();
        if (is_one_hot_label())
        {
            for (int c = 0; c < din_temp.cols(); c++)
            {
                int one_hot_index = one_hot_labels(0, c);
                din_temp(one_hot_index, c) = -1.0 / (input_data(one_hot_index, c) + eps)
                                                / double(input_cols);
                
            }
        }
        else
        {
            for (int r = 0; r < din_temp.rows(); r++)
                for (int c = 0; c < din_temp.cols(); c++)
                    din_temp(r, c) = -normalized_labels(r, c) / (input_data(r, c) + eps)
                                                            / double(input_cols);
        }
        
        din = din_temp / double(input_data.cols());
        //std::cout << input_data << std::endl;
        //std::cout << one_hot_labels << std::endl;
    }
}


