#include <../include/SimpleNN.hpp>

namespace snn
{
    // Simplified LeNet-5 model
    class LeNet : public Sequential
    {
        public:
            LeNet():Sequential()
            {
                int input_img_rows1 = 28;
                int input_img_cols1 = 28;
                int input_img_channels1 = 1;

                int conv_output_img_channels1 = 6;
                int conv_filter_rows1 = 5;
                int conv_filter_cols1 = 5;
                int conv_row_pads1 = 0;
                int conv_col_pads1 = 0;
                int conv_row_strides1 = 1;
                int conv_col_strides1 = 1;

                std::shared_ptr<Convolution> conv_layer1(new Convolution(input_img_rows1, input_img_cols1, 
                                                            input_img_channels1, 
                                                            conv_output_img_channels1, 
                                                            conv_filter_rows1, conv_filter_cols1, 
                                                            conv_row_pads1, conv_col_pads1, 
                                                            conv_row_strides1, conv_col_strides1, 
                                                            0, 0.283, 
                                                            0, 0.01));

                int pool_input_img_rows1 = conv_layer1->output_img_rows;
                int pool_input_img_cols1 = conv_layer1->output_img_cols;
                int pool_filter_rows1 = 2;
                int pool_filter_cols1 = 2;
                int pool_pads1 = 0;
                int pool_strides1 = 2;

                std::shared_ptr<MaxPooling> pool_layer1(new MaxPooling(pool_input_img_rows1, pool_input_img_cols1, 
                                                        pool_filter_rows1, pool_filter_cols1, 
                                                        pool_pads1, pool_pads1, 
                                                        pool_strides1, pool_strides1, 
                                                        conv_output_img_channels1, false));

                int input_img_rows2 = pool_layer1->output_img_rows;
                int input_img_cols2 = pool_layer1->output_img_rows;
                int input_img_channels2 = pool_layer1->image_channels;

                int conv_output_img_channels2 = 16;
                int conv_filter_rows2 = 5;
                int conv_filter_cols2 = 5;
                int conv_row_pads2 = 0;
                int conv_col_pads2 = 0;
                int conv_row_strides2 = 1;
                int conv_col_strides2 = 1;

                std::shared_ptr<Convolution> conv_layer2(new Convolution(input_img_rows2, input_img_cols2, 
                                                            input_img_channels2, 
                                                            conv_output_img_channels2, 
                                                            conv_filter_rows2, conv_filter_cols2, 
                                                            conv_row_pads2, conv_col_pads2, 
                                                            conv_row_strides2, conv_col_strides2, 
                                                            0, 0.115, 
                                                            0, 0.01));

                int pool_input_img_rows2 = conv_layer2->output_img_rows;
                int pool_input_img_cols2 = conv_layer2->output_img_cols;
                int pool_filter_rows2 = 2;
                int pool_filter_cols2 = 2;
                int pool_pads2 = 0;
                int pool_strides2 = 2;

                std::shared_ptr<MaxPooling> pool_layer2(new MaxPooling(pool_input_img_rows2, pool_input_img_cols2, 
                                                        pool_filter_rows2, pool_filter_cols2, 
                                                        pool_pads2, pool_pads2, 
                                                        pool_strides2, pool_strides2, 
                                                        conv_output_img_channels2, true));

                int aff1_input_rows = pool_layer2->output_rows * conv_output_img_channels2; // because flatten-flag is true
                int aff1_input_cols = 1;
                int aff1_output_rows = 120;
                int aff1_output_cols = 1;

                std::shared_ptr<Affine> aff1_layer(new Affine(aff1_input_rows, aff1_input_cols, 
                                                aff1_output_rows, aff1_output_cols, 0, 2.0 / double(aff1_input_rows), 
                                                                                    0, 0.01));

                int aff2_input_rows = 120;
                int aff2_input_cols = 1;
                int aff2_output_rows = 84;
                int aff2_output_cols = 1;

                std::shared_ptr<Affine> aff2_layer(new Affine(aff2_input_rows, aff2_input_cols, 
                                                aff2_output_rows, aff2_output_cols, 0, 2.0 / 120.0, 0, 0.01));

                int aff3_input_rows = 84;
                int aff3_input_cols = 1;
                int aff3_output_rows = 10;
                int aff3_output_cols = 1;

                std::shared_ptr<Affine> aff3_layer(new Affine(aff3_input_rows, aff3_input_cols, 
                                                aff3_output_rows, aff3_output_cols, 0, 2.0 / 84.0, 0, 0.01));

                std::shared_ptr<Relu> relu_layer1(new Relu);
                std::shared_ptr<Relu> relu_layer2(new Relu);
                std::shared_ptr<Relu> relu_layer3(new Relu);
                std::shared_ptr<Relu> relu_layer4(new Relu);
                //std::shared_ptr<Softmax> softmax_layer(new Softmax);

                *this << conv_layer1 << relu_layer1 << pool_layer1
                    << conv_layer2 << relu_layer2 << pool_layer2
                    << aff1_layer << relu_layer3
                    << aff2_layer << relu_layer4
                    <<aff3_layer;
            }
    };
}


