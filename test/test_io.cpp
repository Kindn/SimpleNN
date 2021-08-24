#include <functions/functions.hpp>
#include <Matrix/Matrix.hpp>
#include <functions/common.hpp>
#include <functions/io.hpp>
#include <nets/Dataset.hpp>
#include <iostream>

using namespace snn;

const std::string train_data_file_path = "/home/lpy/SimpleNN/experiments/mnist/dataset/images_train.csv";
const std::string test_data_file_path = "/home/lpy/SimpleNN/experiments/mnist/dataset/images_test.csv";
const std::string train_label_file_path = "/home/lpy/SimpleNN/experiments/mnist/dataset/labels_train.csv";
const std::string test_label_file_path = "/home/lpy/SimpleNN/experiments/mnist/dataset/labels_test.csv";


int main()
{
    /*
    Matrix_d images_train, images_test;
    Matrix_d labels_train, labels_test;

    csv2mat("/home/lpy/SimpleNN/experiments/mnist/dataset/images_train.csv", 
            " ", 
            images_train);
    csv2mat("/home/lpy/SimpleNN/experiments/mnist/dataset/images_test.csv", 
            " ", 
            images_test);
    csv2mat("/home/lpy/SimpleNN/experiments/mnist/dataset/labels_train.csv", 
            " ", 
            labels_train);
    csv2mat("/home/lpy/SimpleNN/experiments/mnist/dataset/labels_test.csv", 
            " ", 
            labels_test);
    
    std::cout << "images_train: " << images_train.rows();
    std::cout << ", " << images_train.cols() << std::endl;
    std::cout << "images_test: " << images_test.rows();
    std::cout << ", " << images_test.cols() << std::endl;
    std::cout << "labels_train: " << labels_train.rows();
    std::cout << ", " << labels_train.cols() << std::endl;
    std::cout << "labels_test: " << labels_test.rows();
    std::cout << ", " << labels_test.cols() << std::endl;*/
    

   /*
    Matrix_d mat;
    if(!csv2mat<double>("/home/lpy/SimpleNN/test/test_io.csv", 
            " ", 
            mat))
        std::cout << "Fail" << std::endl;
    std::cout << mat << std::endl;*/
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

    Matrix_d mini_batch_data;
    Label mini_batch_labels(true);

    train_set.random_choose(mini_batch_data, mini_batch_labels, 100);
    std::cout << mini_batch_data.rows() << " " << mini_batch_data.cols() << std::endl;


    return 0;
}
