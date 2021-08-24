/*
 * filename: Label.hpp
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    Declaration of classes that discribe training labels.
 */

#ifndef  _LABEL_HPP_
#define  _LABEL_HPP_

#include <Matrix/Matrix.hpp>
#include <functions/io.hpp>
#include <sstream>
#include <iostream>
#include <string>
#include <memory>

namespace snn
{
    class Label : public std::enable_shared_from_this<Label>
    {
        private:
            Matrix_d normal_labels;
            Matrix_i one_hot_labels;

            bool one_hot_flag;
            int size;

        public:
            Label()
            {
                one_hot_flag = false;
                size = 0;
            }
            Label(bool _one_hot_flag):one_hot_flag(_one_hot_flag)
            {this->size = 0;}
            ~Label() {}

            bool is_one_hot_label() {return one_hot_flag;}

            void set_normal_labels(Matrix_d _nl)
            {
                if (one_hot_flag)
                {
                    std::stringstream msg;
                    msg << "[Snn Error]In Label::set_normal_labels:";
                    msg << "Label is one-hot,you should not set normal labels"<< std::endl;
                    throw msg.str();
                }
                else 
                {
                    this->normal_labels = _nl;
                    this->size = _nl.cols();
                }           
            }

            void set_one_hot_labels(Matrix_i _ohl)
            {
                if (!one_hot_flag)
                {
                    std::stringstream msg;
                    msg << "[Snn Error]In Label::set_one_hot_labels:";
                    msg << "Label is normal,you should not set one-hot labels"<< std::endl;
                    throw msg.str();
                }
                else 
                {
                    this->one_hot_labels = _ohl;
                    this->size = _ohl.cols();
                }         
            }
            
            Matrix_d get_normal_labels()
            {

                if (one_hot_flag)
                {
                    std::stringstream msg;
                    msg << "[Snn Error]In Label::get_normal_labels:";
                    msg << "Label is one-hot,you should not get normal labels"<< std::endl;
                    throw msg.str();
                }
                else return this->normal_labels;
            }

            Matrix_i get_one_hot_labels()
            {
                if (!one_hot_flag)
                {
                    std::stringstream msg;
                    msg << "[Snn Error]In Label::get_one_hot_labels:";
                    msg << "Label is normal,you should not get one-hot labels"<< std::endl;
                    throw msg.str();
                }
                else return this->one_hot_labels;
            }

            int get_size() const
            {return this->size;}

            bool load_label(const std::string& _path)
            {
                if (this->one_hot_flag)
                {
                    if (csv2mat<int>(_path, " ", this->one_hot_labels))
                    {
                        this->size = this->one_hot_labels.cols();
                        return true;
                    }
                    else
                        return false;
                }
                else
                {
                    if (csv2mat<double>(_path, " ", this->normal_labels))
                    {
                        this->size = this->normal_labels.cols();
                        return true;
                    }
                    else 
                        return false;
                }
            }
    };
}


#endif //_LABEL_HPP_
