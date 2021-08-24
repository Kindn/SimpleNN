/*
 * filename: Dataset.hpp
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    Declaration of dataset class.
 */

#ifndef  _DATASET_HPP_
#define  _DATASET_HPP_

#include <Matrix/Matrix.hpp>
#include <nets/Label.hpp>
#include <sstream>
#include <string>
#include <vector>
#include <iostream>
#include <memory>
#include <ctime>

namespace snn
{
    class Dataset : public std::enable_shared_from_this<Dataset>
    {
        private:
            Matrix_d data;
            Label labels;

            bool one_hot_flag;
            int data_set_size;

        public:
            Dataset()
            {
                one_hot_flag = false;
                data_set_size = 0;
            }
            Dataset(bool _one_hot_flag)
            {
                one_hot_flag = _one_hot_flag;
                data_set_size = 0;
                Label tmp(_one_hot_flag);
                labels = tmp;
            }
            Dataset(bool _one_hot_flag, Matrix_d& _data, Label& _labels);
            virtual ~Dataset() {}

            Matrix_d get_data() const
            {return this->data;}
            Label get_labels() const
            {return this->labels;}
            int get_size() const 
            {return this->data_set_size;}

            bool labels_are_one_hot() const
            {return this->one_hot_flag;}

            virtual bool load_data(const std::string& _data_file_path, 
                                const std::string& _label_file_path);

            virtual void random_choose(Matrix_d&, Label&, int);
    };
}




#endif //_DATASET_HPP_
