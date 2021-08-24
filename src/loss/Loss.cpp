/*
 * filename: Loss.cpp
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    Definition of basic class Loss's methods.
 */

#include <loss/Loss.hpp>

namespace snn
{
    void Loss::set_labels(Matrix_d _labels)
    {
        if (one_hot_flag)
        {
            std::stringstream msg;
            msg << "Set labels failed!Need normalized labels!\n";
            throw msg.str();
        }
        else
        {
            normalized_labels = _labels;
        }
    }

    void Loss::set_labels(Matrix_i _labels)
    {
        if (!one_hot_flag)
        {
            std::cerr << "Set labels failed!Need one-hot labels!" << std::endl;
            return;
        }
        else
        {
            one_hot_labels = _labels;
        }
    }

    double Loss::accuracy() const
    {
        if (!one_hot_flag)
        {
            std::stringstream msg;
            msg << "In Loss::accuracy: ";
            msg << "The labels are not one-hot.Cannot calculate accuracy\n";
            throw msg.str();
        }
        else
        {
            std::vector<int> max_pos;
            argmax_axis<double>(input_data, max_pos, 1);
            int correct_cnt = 0;
            for (int i = 0; i < max_pos.size(); i++)
            {
                if (max_pos[i] == one_hot_labels(0, i))
                    correct_cnt++;
            }

            return double(correct_cnt) / double(one_hot_labels.cols());
        }
    }
}
