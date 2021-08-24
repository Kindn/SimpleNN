/*
 * filename: Sequential.hpp
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    Declaration of sequential network.
 */

#ifndef  _SEQUENTIAL_HPP_
#define  _SEQUENTIAL_HPP_

#include <Matrix/Matrix.hpp>
#include <Matrix/exceptions.hpp>

#include <functions/functions.hpp>

#include <layers/Layer.hpp>
#include <layers/Layer.hpp>
#include <layers/Affine.hpp>
#include <layers/Convolution.hpp>
#include <layers/LRelu.hpp>
#include <layers/MaxPooling.hpp>
#include <layers/Relu.hpp>
#include <layers/Sigmoid.hpp>
#include <layers/Softmax.hpp>

#include <loss/Loss.hpp>

#include <optimizers/Optimizer.hpp>

#include <nets/Label.hpp>
#include <nets/Dataset.hpp>

#include <data/json.hpp>

#include <vector>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>

namespace snn
{
    class Sequential : public std::enable_shared_from_this<Sequential>
    {
        private:
            std::vector<std::shared_ptr<Layer>> layers;
            std::shared_ptr<Loss> loss_layer;
            std::shared_ptr<Optimizer> opt_ptr;

            std::vector<Matrix_d*> params;
            std::vector<Matrix_d*> grads;
            double loss;
            

            Matrix_d input_data;
            Matrix_d output_data;
            Matrix_d din;

        public:
            Sequential() {}
            virtual ~Sequential() {}

            void add_layer(std::shared_ptr<Layer> _layer);

            Sequential& operator << (std::shared_ptr<Layer> _layer)
            {
                this->add_layer(_layer);
                return *this;
            }

            void set_loss_layer(std::shared_ptr<Loss> _loss_layer);

            void set_optimizer(std::shared_ptr<Optimizer> _opt_ptr);

            int num_layers() const;

            std::vector<Matrix_d> get_params() const;

            std::vector<Matrix_d> get_grads() const;

            void set_input(const Matrix_d& _input_data);

            Matrix_d get_output() const;
            
            void update();

            Matrix_d predict(const Matrix_d& x);

            double get_loss() const;

            virtual void fit(Dataset& train_set, Dataset& test_set, 
                            const int _batch_size, const int _epoches);

            virtual bool save_net(const std::string& _export_file_path);

            virtual bool save_weights(const std::string& _export_file_path);

            virtual bool save_model(const std::string& _export_net_file_path, 
                                    const std::string& _export_weight_file_path);

            virtual bool load_net(const std::string& _net_file_path);

            virtual bool load_weights(const std::string& _weight_file_path);

            virtual bool load_model(const std::string& _net_file_path, 
                                    const std::string& _weight_file_path);
            
        protected:
            void forward_propagation();

            void backward_propagation();

            void set_loss_labels(Label& _labels);

    };
}



#endif //_SEQUENTIAL_HPP_
