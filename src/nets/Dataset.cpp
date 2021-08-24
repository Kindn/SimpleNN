/*
 * filename: Dataset.cpp
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    Definition of Dataset's methods.
 */

#include <nets/Dataset.hpp>
#include <functions/io.hpp>

namespace snn
{
    Dataset::Dataset(bool _one_hot_flag, Matrix_d& _data, Label& _labels)
    {
        if (_one_hot_flag != _labels.is_one_hot_label())
        {
            std::stringstream msg;
            msg << "In Dataset::Dataset: ";
            msg << "Label types mismatch!";
            throw msg.str();
        }
        if (_data.cols() != _labels.get_size())
        {
            std::stringstream msg;
            msg << "In Dataset::Dataset: ";
            msg << "Data sizes mismatch!";
            throw msg.str();
        }
        
        one_hot_flag = _one_hot_flag;
        data_set_size = _data.cols();
        data = _data;
        labels = _labels;
    }

    bool Dataset::load_data(const std::string& _data_file_path, 
                            const std::string& _label_file_path)
    {
        if (!csv2mat<double>(_data_file_path, " ", this->data)) 
            return false;
        if (!this->labels.load_label(_label_file_path)) 
            return false;
        if (this->data.cols() != this->labels.get_size()) 
            return false;
        this->data_set_size = this->data.cols();
        return true;
    }

    void Dataset::random_choose(Matrix_d& _mini_batch_data, 
                                Label& _mini_batch_labels,
                                int _mini_batch_size)
    {
        if (this->one_hot_flag != _mini_batch_labels.is_one_hot_label())
        {
            std::stringstream msg;
            msg << "In Sequential::set_loss_label: ";
            msg << "Label type mismatch!\n";
            throw msg.str();
        }

        int col_upper = this->data_set_size;
        if (_mini_batch_size > col_upper)
        {
            std::stringstream msg;
            msg << "In Dataset::random_choose: ";
            msg << "_mini_batch_size is larger than data size!\n";
            throw msg.str();
        }
        else if (_mini_batch_size == this->data_set_size)
        {
            _mini_batch_data = this->data;
            if (this->one_hot_flag)
            {
                Matrix_i mini_batch_labels;
                mini_batch_labels = this->labels.get_one_hot_labels();
                _mini_batch_labels.set_one_hot_labels(mini_batch_labels);
            }
            else
            {
                Matrix_d mini_batch_labels;
                mini_batch_labels = this->labels.get_normal_labels();
                _mini_batch_labels.set_normal_labels(mini_batch_labels);
            }
        }
        else
        {
            std::vector<int> row_set;
            row_set.clear();
            for (int i = 0; i < this->data.rows(); i++)
                row_set.push_back(i);
            std::vector<int> col_set;
            col_set.clear();
            int *chosen_flags = new int[col_upper]();
            srand(time(NULL));
            for (int i = 0; i < _mini_batch_size; i++)
            {
                int c;
                int cnt = 0;
                do
                {
                    if (cnt >= col_upper)
                    {
                        for (int j = 0; j < col_upper; j++)
                        {
                            if (chosen_flags[j] == 0)
                            {
                                c = j;
                                break;
                            }
                        }
                        break;
                    }
                    c = rand() % col_upper;
                    cnt++;
                } while (chosen_flags[c] != 0);
                col_set.push_back(c);
                chosen_flags[c] = 1;
            }

            delete [] chosen_flags;

            _mini_batch_data = this->data.sub_matrix(row_set, col_set);
            if (this->one_hot_flag)
            {
                Matrix_i mini_batch_labels;
                row_set.clear();
                row_set.push_back(0);
                mini_batch_labels = this->labels.get_one_hot_labels().sub_matrix(row_set, col_set);
                _mini_batch_labels.set_one_hot_labels(mini_batch_labels);
            }
            else
            {
                Matrix_d mini_batch_labels;
                mini_batch_labels = this->labels.get_normal_labels().sub_matrix(row_set, col_set);
                _mini_batch_labels.set_normal_labels(mini_batch_labels);
            }
        }
        
    }
}


