#include <layers/Affine.hpp>
#include <layers/Relu.hpp>
#include <layers/Convolution.hpp>
#include <layers/MaxPooling.hpp>
#include <functions/functions.hpp>

#include <iostream>

using namespace std;
using namespace snn;

int main()
{
    /*
    int input_data_rows = 144, input_data_cols = 300;
    Matrix_d input_data = nrandom(input_data_rows, input_data_cols);

    cout << "input_data = " << endl;
    cout << input_data << endl;

    // forward propagation
    Convolution conv_layer(12, 12, 3, 3, 3, 3, 1, 1, 2, 2);
    conv_layer.set_input(input_data);
    conv_layer.forward();
    Matrix_d conv_output = conv_layer.get_output();
    cout << "conv_output = " << endl;
    cout << conv_output << endl;

    MaxPooling pool_layer(conv_layer.output_img_rows, conv_layer.output_img_cols, 
                          2, 2, 1, 1, 2, 2, 3);
    pool_layer.set_input(conv_output);
    pool_layer.forward();
    Matrix_d pool_output = pool_layer.get_output();
    cout << "pool_output = " << endl;
    cout << pool_output << endl;

    Affine affine_layer(pool_output.rows(), 1, 10, 1);
    affine_layer.set_input(pool_output);
    affine_layer.forward();
    Matrix_d affine_output = affine_layer.get_output();
    cout << "affine_output = " << endl;
    cout << affine_output << endl;

    Relu relu_layer;
    relu_layer.set_input(affine_output);
    relu_layer.forward();
    Matrix_d relu_output = relu_layer.get_output();
    cout << "relu_output = " << endl;
    cout << relu_output << endl;

    SoftmaxLoss smxloss_layer(relu_output.rows(), 1, false);
    Matrix_d labels = urandom(relu_output.rows(), relu_output.cols());
    //Matrix_d labels = urandom(relu_output.rows(), relu_output.cols(), 0, relu_output.rows()).cast<int>();
    smxloss_layer.set_input(relu_output);
    smxloss_layer.set_labels(labels);
    smxloss_layer.forward();
    Matrix_d smxloss_output = smxloss_layer.get_output();
    cout << "smxloss_output = " << endl;
    cout << smxloss_output << endl;

    // backward propagation
    Matrix_d dLoss = eye<double>(1) * 0.5;

    smxloss_layer.backward(dLoss);
    cout << "dSoftmaxLossInput = " << endl;
    cout << smxloss_layer.din << endl;
    relu_layer.backward(smxloss_layer.din);
    cout << "dReluInput = " << endl;
    cout << relu_layer.din << endl;
    affine_layer.backward(relu_layer.din);
    cout << "dAffineInput = " << endl;
    cout << affine_layer.din << endl;
    cout << "dAffineWeights = " << endl;
    cout << affine_layer.dWeights << endl;
    cout << "dAffineBias = " << endl;
    cout << affine_layer.dBias << endl;
    pool_layer.backward(affine_layer.din);
    cout << "dPoolingInput = " << endl;
    cout << pool_layer.din << endl;
    conv_layer.backward(pool_layer.din);
    cout << "dConvInput = " << endl;
    cout << conv_layer.din << endl;
    cout << "dConvFilter = " << endl;
    cout << conv_layer.dFilter << endl;
    cout << "dConvBias = " << endl;
    cout << conv_layer.dBias << endl;
    */

    double a7[16] = {0, 2, 3, 4, 5, 6, 7, 8, 6, 2, 5, 10, 2, 4, 6, 6};
    Matrix_d input(a7, 16, 1);

    std::cout << "input = " << input.reshape(4, 4) << std::endl;

    MaxPooling pool_layer(4, 4, 2, 2, 0, 0, 2, 2, 1);
    pool_layer.set_input(input);
    pool_layer.forward();
    std::cout << "output = " << pool_layer.get_output().reshape(2, 2) << std::endl;

    Matrix_d dout = urandom(4, 1);
    std::cout << "dout = " << dout.reshape(2, 2);

    pool_layer.backward(dout);
    std::cout << "din = " << pool_layer.din.reshape(4, 4) << std::endl;

    return 0;
}

