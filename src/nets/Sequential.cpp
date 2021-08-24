/*
 * filename: Sequential.cpp
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    Definition of sequential network's methods.
 */

#include <nets/Sequential.hpp>

namespace snn
{
    void Sequential::add_layer(std::shared_ptr<Layer> _layer)
    {
        this->layers.push_back(_layer->shared_from_this());
        _layer->get_params_ptr(this->params, this->grads);
    }

    void Sequential::set_loss_layer(std::shared_ptr<Loss> _loss_layer)
    {
        this->loss_layer = _loss_layer->shared_from_this();
    }

    void Sequential::set_optimizer(std::shared_ptr<Optimizer> _opt_ptr)
    {
        this->opt_ptr = _opt_ptr->shared_from_this();
    }

    void Sequential::set_loss_labels(Label& _label)
    {
        if (_label.is_one_hot_label() != this->loss_layer->is_one_hot_label())
        {
            std::stringstream msg;
            msg << "In Sequential::set_loss_label: ";
            msg << "Label type mismatch!\n";
            throw msg.str();
        }
        else
        {
            if (_label.is_one_hot_label())
            {
                Matrix_i ohl;
                ohl = _label.get_one_hot_labels();
                this->loss_layer->set_labels(ohl);
            }
            else
            {
                Matrix_d nl;
                nl = _label.get_normal_labels();
                this->loss_layer->set_labels(nl);
            }
        }
    }

    int Sequential::num_layers() const
    {
        return this->layers.size();
    }

    std::vector<Matrix_d> Sequential::get_params() const
    {
        std::vector<Matrix_d> result;
        for (int i = 0; i < this->params.size(); i++)
        {
            Matrix_d param_tmp = *(params[i]);
            result.push_back(param_tmp);
        }
        return result;
    }

    std::vector<Matrix_d> Sequential::get_grads() const
    {
        std::vector<Matrix_d> result;
        for (int i = 0; i < this->grads.size(); i++)
        {
            Matrix_d grad_tmp = *(grads[i]);
            result.push_back(grad_tmp);
        }
        return result;
    }

    void Sequential::set_input(const Matrix_d& _input_data)
    {
        this->input_data = _input_data;
    }

    Matrix_d Sequential::get_output() const
    {
        return this->output_data;
    }

    void Sequential::forward_propagation()
    {
        Matrix_d output_temp;

        if (this->num_layers() <= 0)
            return;
        else
        {
            // propagation in first layer
            std::shared_ptr<Layer> first_layer = this->layers[0];
            first_layer->set_input(this->input_data);
            first_layer->forward();
            output_temp = first_layer->get_output();

            // propagation in hidden layers and output layer
            for (int layer_index = 1; layer_index < this->layers.size();
                                    layer_index++)
            {
                this->layers[layer_index]->set_input(output_temp);
                this->layers[layer_index]->forward();
                output_temp = this->layers[layer_index]->get_output();
            }

            this->output_data = output_temp;
            
            // propagation in loss layer
            this->loss_layer->set_input(output_temp);
            this->loss_layer->forward();
            this->loss = this->loss_layer->get_loss();
        }
        
    }

    void Sequential::backward_propagation()
    {
        Matrix_d din_temp;

        if (this->num_layers() <= 0)
            return;
        else
        {
            // propagation in loss layer
            this->loss_layer->backward();
            din_temp = this->loss_layer->din;
            // propagation in other layers
            for (int layer_index = this->layers.size() - 1; layer_index >= 0; 
                                                            layer_index--)
            {
                this->layers[layer_index]->backward(din_temp);
                din_temp = this->layers[layer_index]->din;
            }

            this->din = din_temp;
        }
        
    }

    void Sequential::update()
    {
        if (this->params.size() <= 0)
            return;
        else
        {
            this->opt_ptr->update(this->params, this->grads);
        }
    }

    Matrix_d Sequential::predict(const Matrix_d& x)
    {
        if (this->params.size() <= 0)
            return x;
        else
        {
            Matrix_d result;
            std::shared_ptr<Layer> first_layer = this->layers[0];

            first_layer->set_input(x);
            first_layer->forward();
            result = first_layer->get_output();
            //std::cout << result << std::endl;
            for (int layer_index = 1; layer_index < this->layers.size();
                                        layer_index++)
            {
                this->layers[layer_index]->set_input(result);
                this->layers[layer_index]->forward();
                result = this->layers[layer_index]->get_output();
                //std::cout << result << std::endl;
            }

            this->output_data = result;
            return result;
        }
        
    }

    double Sequential::get_loss() const
    {
        return this->loss;
    }

    void Sequential::fit(Dataset& train_set, 
                        Dataset& test_set, 
                        const int _batch_size, 
                        const int _epoches)
    { 
        if ((train_set.labels_are_one_hot() != 
            test_set.labels_are_one_hot()) || 
            (train_set.labels_are_one_hot() != 
            this->loss_layer->is_one_hot_label()))
        {
            std::stringstream msg;
            msg << "In Sequential::fit:";
            msg << "Label types mismatch!";
            throw msg.str();
        }
        const int data_size = train_set.get_size();
        if (_batch_size > data_size)
        {
            std::stringstream msg;
            msg << "In Sequential::fit:";
            msg << "_batch_size should not be larger than data size!";
            throw msg.str();
        }
        const int iter_per_epoch = data_size / _batch_size;

        std::cout << "Training start: " << std::endl;
        std::cout << "---------------------------------------------" << std::endl;
        for (int iter_num = 0; iter_num < _epoches * iter_per_epoch;
                            iter_num++)
        {
            std::cout << "iter_num = " << iter_num << std::endl;

            double train_loss, test_loss;
            Matrix_d mini_batch_data;

            if (train_set.labels_are_one_hot())
            {
                Label mini_batch_labels(true);
                Label test_labels(true);
                Matrix_d test_data;
                train_set.random_choose(mini_batch_data, 
                                        mini_batch_labels, 
                                        _batch_size);
                test_set.random_choose(test_data, 
                                    test_labels, 
                                    100);
            
                this->loss_layer->set_labels(test_labels.get_one_hot_labels());
                this->input_data = test_data;
                this->forward_propagation();
                double test_accuracy = this->loss_layer->accuracy();
                test_loss = this->loss;

                this->loss_layer->set_labels(mini_batch_labels.get_one_hot_labels());
                this->input_data = mini_batch_data;
                this->forward_propagation();
                double train_accuracy = this->loss_layer->accuracy();
                train_loss = this->loss;
                this->backward_propagation();
                this->update();
                
                std::cout << "train_loss = " << train_loss << ", ";
                std::cout << "test_loss = " << test_loss << std::endl;
                std::cout << "train_accuracy = " << train_accuracy << ", ";
                std::cout << "test_accuracy = " << test_accuracy << std::endl;
                std::cout << std::endl;
            }
            else
            {
                Label mini_batch_labels(false);
                Label test_labels(false);
                Matrix_d test_data;
                train_set.random_choose(mini_batch_data, 
                                        mini_batch_labels, 
                                        _batch_size);
                test_set.random_choose(test_data, 
                                    test_labels, 
                                    10);
                this->loss_layer->set_labels(test_labels.get_normal_labels());
                this->input_data = test_data;
                this->forward_propagation();
                test_loss = this->loss;

                this->loss_layer->set_labels(mini_batch_labels.get_normal_labels());
                this->input_data = mini_batch_data;
                this->forward_propagation();
                train_loss = this->loss;
                this->backward_propagation();
                this->update();

                std::cout << "train_loss = " << train_loss << ", ";
                std::cout << "test_loss = " << test_loss << std::endl;
            }
            
        }
        std::cout << "Fitting Done!" << std::endl;
    }

    bool Sequential::save_net(const std::string& _export_file_path)
    {
        sjson::Json net_file;
        sjson::JsonNode& root = net_file.getRoot();
        try
        {
            root["net_type"] = "Sequential";
            if (root["layers"].set_array(num_layers()))
            {
                for (int i = 0; i < num_layers(); i++)
                {
                    std::string layer_type;
                    LayerParams layer_params;
                    if (!layers[i]->get_properties(layer_params, 
                                            layer_type))
                        return false;
                    else
                    {
                        root["layers"][i]["layer_type"] = layer_type;
                        for (LayerParams::iterator it = layer_params.begin(); 
                            it != layer_params.end(); it++)
                        {
                            std::string key = it->first;
                            double value = it->second;
                            root["layers"][i][key] = value;
                        }
                    }
                    
                }
            }
            else
                return false;
            
            if(!net_file.save(_export_file_path, true))
                return false;
        }
        catch(const std::exception& e)
        {
            std::cerr << e.what() << '\n';
            return false;
        }

        return true;
    }

    bool Sequential::save_weights(const std::string& _export_file_path)
    {
        std::ofstream ofs(_export_file_path, 
                        std::ios::out | std::ios::binary);
        if (ofs.fail())
        {
            ofs.close();
            return false;
        }

        for (int i = 0; i < this->params.size(); i++)
        {
            if (ofs.good())
            {
                char *data = (char *)(params[i]->get_data());
                int out_bytes = params[i]->size() * sizeof(double);
                ofs.write(data, out_bytes);
            }
            else
            {
                ofs.close();
                return false;
            }
        }

        ofs.close();
        return true;
    }

    bool Sequential::save_model(const std::string& _export_net_file_path, 
                                const std::string& _export_weight_file_path)
    {
        if (!this->save_net(_export_net_file_path))
            return false;
        if (!this->save_weights(_export_weight_file_path))
            return false;

        return true;
    }

    bool Sequential::load_net(const std::string& _net_file_path)
    {
        this->layers.clear();
        std::ifstream ifs(_net_file_path);
        sjson::Json net_file(ifs);
        if (net_file.fail())
        {
            std::cerr << "In Sequential::load_net: ";
            std::cerr << "cannot open net config file " << _net_file_path;
            std::cerr << std::endl;
            ifs.close();
            return false;
        }

        sjson::JsonNode& root = net_file.getRoot();
        if (root.obj_has_item("net_type") && root.obj_has_item("layers"))
        {
            std::string config_net_type = root["net_type"].as_string();
            if (config_net_type != "Sequential")
            {
                std::cerr << "In Sequential::load_net: ";
                std::cerr << "net type mismatched.";
                std::cerr << "config type: " << config_net_type << ";";
                std::cerr << "required type: " << "Sequential.";
                std::cerr << std::endl;
                ifs.close();
                return false;
            }
            else
            {
                sjson::JsonNode& config_layers = root["layers"];
                int config_nlayer = config_layers.get_children_size();
                for (int i = 0; i < config_nlayer; i++)
                {
                    std::string layer_type = config_layers[i]["layer_type"].as_string();
                    std::vector<sjson::JsonNode_P> config_layer_items = config_layers[i].as_vector();
                    LayerParams config_layer_params;
                    for (int j = 0; j < config_layer_items.size(); j++)
                    {
                        std::string key = config_layer_items[j]->get_key();
                        if (key.substr(0, 6) == "param_")
                        {
                            int value_type = config_layer_items[j]->get_value_type();
                            if (value_type == sjson::JSON_VALUE_TYPE_INT || 
                                value_type == sjson::JSON_VALUE_TYPE_DOUBLE || 
                                value_type == sjson::JSON_VALUE_TYPE_BOOL)
                            {
                                double val = config_layer_items[j]->as_double();
                                config_layer_params.insert(LayerParams::value_type(key, val));
                            }
                        }
                    }

                    std::shared_ptr<Layer> new_layer;
                    if (layer_type == "Affine")
                        new_layer = std::make_shared<Affine>();
                    else if (layer_type == "Convolution")
                        new_layer = std::make_shared<Convolution>();
                    else if (layer_type == "LRelu")
                        new_layer = std::make_shared<LRelu>();
                    else if (layer_type == "MaxPooling")
                        new_layer = std::make_shared<MaxPooling>();
                    else if (layer_type == "Relu")
                        new_layer = std::make_shared<Relu>();
                    else if (layer_type == "Sigmoid")
                        new_layer = std::make_shared<Sigmoid>();
                    else if (layer_type == "Softmax")
                        new_layer = std::make_shared<Softmax>();
                    else
                    {
                        std::cerr << "In Sequential::load_net: ";
                        std::cerr << "unsurpported layer type: " << layer_type;
                        std::cerr << std::endl;
                        ifs.close();
                        return false;
                    }

                    if (!new_layer->set_properties(config_layer_params))
                    {
                        ifs.close();
                        return false;
                    }
                    else
                        this->add_layer(new_layer);
                }
            }
        }
        else
        {
            std::cerr << "In Sequential::load_net: ";
            std::cerr << "invalid config file " << _net_file_path;
            std::cerr << std::endl;
            return false;
        }

        return true;
    }

    bool Sequential::load_weights(const std::string& _weight_file_path)
    {
        std::ifstream ifs(_weight_file_path, 
                        std::ios::in | std::ios::binary);
        if (ifs.fail())
        {
            ifs.close();
            return false;
        }

        int param_bytes = 0;
        for (int i = 0; i < this->params.size(); i++)
            param_bytes += params[i]->size() * sizeof(double);
        ifs.seekg(0, ifs.end);
        int total_in_bytes = ifs.tellg();
        if (total_in_bytes != param_bytes)
        {
            std::cerr << "In Sequential::load_weights: ";
            std::cerr << "weight size mismatched!";
            std::cerr << std::endl;
            ifs.close();
            return false;
        }
        else
        {
            ifs.seekg(0, ifs.beg); // necessary!
            for (int i = 0; i < this->params.size(); i++)
            {
                if (ifs.good())
                {
                    char *data = (char *)(params[i]->get_data());
                    int in_bytes = params[i]->size() * sizeof(double);
                    ifs.read(data, in_bytes);
                }
                else
                {
                    ifs.close();
                    return false;
                }
            }
        }

        ifs.close();
        return true;
    }

    bool Sequential::load_model(const std::string& _net_file_path, 
                                const std::string& _weight_file_path)
    {
        if (!this->load_net(_net_file_path))
            return false;
        if (!this->load_weights(_weight_file_path))
            return false;
        
        return true;
    }
}


