/*
 * filename: Loss.hpp
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    Declaration of basic class Loss.
 */

#ifndef  _LOSS_HPP_
#define  _LOSS_HPP_

#include <Matrix/Matrix.hpp>
#include <functions/functions.hpp>
#include <functions/common.hpp>
#include <vector>
#include <sstream>
#include <memory>

namespace snn
{
    class Loss : public std::enable_shared_from_this<Loss>
    {
        public:
            Matrix_d input_data;
            Matrix_d din;
            Matrix_d normalized_labels;
            Matrix_i one_hot_labels;
            double loss;
        private:
            bool one_hot_flag;

        public:
            Loss(bool _one_hot_flag) {one_hot_flag = _one_hot_flag;}

            void set_labels(Matrix_d _labels);
            void set_labels(Matrix_i _labels);

            void set_input(Matrix_d _input_data)
            {
                input_data = _input_data;
            }

            double get_loss() {return loss;}
            bool is_one_hot_label() {return one_hot_flag;}

            double accuracy() const;

            virtual void forward() = 0;
            virtual void backward() = 0;
    };

}


#endif //_LOSS_HPP_
