#include <SimpleNN.hpp>

using namespace snn;

const std::string train_data_file_path = "/home/lpy/SimpleNN/experiments/mnist/dataset/images_train.csv";
const std::string test_data_file_path = "/home/lpy/SimpleNN/experiments/mnist/dataset/images_test.csv";
const std::string train_label_file_path = "/home/lpy/SimpleNN/experiments/mnist/dataset/labels_train.csv";
const std::string test_label_file_path = "/home/lpy/SimpleNN/experiments/mnist/dataset/labels_test.csv";


int main()
{
    int input_img_rows = 28;
    int input_img_cols = 28;
    int input_img_channels = 1;

    int conv_output_img_channels = 10;
    int conv_filter_rows = 5;
    int conv_filter_cols = 5;
    int conv_row_pads = 0;
    int conv_col_pads = 0;
    int conv_row_strides = 1;
    int conv_col_strides = 1;

    std::shared_ptr<Convolution> conv_layer(new Convolution(input_img_rows, input_img_cols, 
                                                input_img_channels, 
                                                conv_output_img_channels, 
                                                conv_filter_rows, conv_filter_cols, 
                                                conv_row_pads, conv_col_pads, 
                                                conv_row_strides, conv_col_strides, 
                                                0, 0.01, 
                                                0, 0.01));

    int pool_input_img_rows = conv_layer->output_img_rows;
    int pool_input_img_cols = conv_layer->output_img_cols;
    int pool_filter_rows = 2;
    int pool_filter_cols = 2;
    int pool_pads = 0;
    int pool_strides = 2;

    std::shared_ptr<MaxPooling> pool_layer(new MaxPooling(pool_input_img_rows, pool_input_img_cols, 
                                            pool_filter_rows, pool_filter_cols, 
                                            pool_pads, pool_pads, 
                                            pool_strides, pool_strides, 
                                            conv_output_img_channels, true));
    
    int aff1_input_rows = pool_layer->output_rows * conv_output_img_channels; // because flatten-flag is true
    int aff1_input_cols = 1;
    int aff1_output_rows = 100;
    int aff1_output_cols = 1;

    std::shared_ptr<Affine> aff1_layer(new Affine(aff1_input_rows, aff1_input_cols, 
                                    aff1_output_rows, aff1_output_cols, 0, 0.1, 0, 0.1));
    
    int aff2_input_rows = aff1_layer->output_rows;
    int aff2_input_cols = 1;
    int aff2_output_rows = 10;
    int aff2_output_cols = 1;

    std::shared_ptr<Affine> aff2_layer(new Affine(aff2_input_rows, aff2_input_cols, 
                                    aff2_output_rows, aff2_output_cols, 0, 0.01, 0, 0.01));
    std::shared_ptr<Relu> relu1_layer(new Relu);
    std::shared_ptr<Relu> relu2_layer(new Relu);
    std::shared_ptr<Softmax> softmax_layer(new Softmax);

    Sequential net;

    net << conv_layer << relu1_layer << pool_layer
        << aff1_layer << relu2_layer
        << aff2_layer << softmax_layer;

    std::shared_ptr<CrossEntropyLoss> loss_layer(new CrossEntropyLoss(true));
    net.set_loss_layer(loss_layer);
    std::cout << "Loss layer ready!" << std::endl;

    std::vector<Matrix_d> init_params = net.get_params();
    std::vector<Matrix_d> init_grads = net.get_grads();
    //std::shared_ptr<AdaGrad> opt(new AdaGrad(init_params, init_grads, 0.0008));
    std::shared_ptr<GradDescent> opt(new GradDescent(init_params, init_grads, 0.02));
    //std::shared_ptr<Momentum> opt(new Momentum(init_params, init_grads, 0.6, 0.01));
    net.set_optimizer(opt);
    std::cout << "Optimizer ready!" << std::endl;
    
    Dataset train_set(true);
    Dataset test_set(true);
    
    if (train_set.load_data(train_data_file_path, train_label_file_path))
        std::cout << "Train set loading finished!" << std::endl;
    else
        std::cout << "Failed to load train set data!" << std::endl;

    if (test_set.load_data(test_data_file_path, test_label_file_path))
        std::cout << "Test set loading finished!" << std::endl;
    else
        std::cout << "Failed to load test set data!" << std::endl;

    net.fit(train_set, test_set, 512, 2);
    
    /*Matrix_d mini_batch_data;
    Label mini_batch_labels(true);
            Label test_labels(true);
            Matrix_d test_data;
            train_set.random_choose(mini_batch_data, 
                                    mini_batch_labels, 
                                    5);
    net.predict(mini_batch_data);*/

    return 0;
}
