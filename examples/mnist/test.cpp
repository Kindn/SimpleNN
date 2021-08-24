#include <SimpleNN.hpp>
#include <LeNet/LeNet.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>
#include <vector>

const std::string weight_path = "../../../examples/mnist/LeNet.weights";

void cvmat2nndata(cv::Mat& _src, Matrix_d& _nndata, bool _reverse)
{
    int rows = _src.rows;
    int cols = _src.cols;
    int channels = _src.channels();

    if (_src.depth() != CV_8U)
        return;

    _nndata = snn::zeros<double>(rows * cols, channels);
    for (int r = 0; r < rows; r++)
    {
        for (int c = 0; c < cols; c++)
        {
            uchar *ptr = _src.ptr<uchar>(r, c);
            for (int ch = 0; ch < channels; ch++)
            {
                if (_reverse)
                    _nndata(r * cols + c, ch) = 255 - ptr[ch];
                else
                    _nndata(r * cols + c, ch) = ptr[ch];
            }
                
        }
    }
}

int main (int argc, char **argv)
{
    if (argc <= 1)
    {
        std::cout << "Please specify an image file path!" << std::endl;
        return -1;
    }
    else
    {
        const std::string path = argv[1];
        bool reverse_flag = false;
        if (argc >= 3)
        {
            std::string argname(argv[2]);
            if (argname == "--reverse")
                reverse_flag = true;
        }
            
        cv::Mat src = cv::imread(path);
        if (src.channels() != 1)
            cv::cvtColor(src, src, cv::COLOR_BGR2GRAY);
        cv::Mat src1;
        cv::resize(src, src1, cv::Size(28, 28));
        cv::imshow("resize", src1);

        Matrix_d input;
        cvmat2nndata(src1, input, reverse_flag);

        snn::LeNet net;
        if (!net.load_weights(weight_path))
        {
            std::cerr << "Failed to load weights!" << std::endl;
            return -1;
        }

        net.predict(input);
        Matrix_d output = net.get_output();
        output = snn::softmax(output, 1);
        std::vector<int> max_cols;
        snn::argmax_axis<double>(output, max_cols, 1);
        int result = max_cols[0];
        double softmax = output(result, 0);
        std::cout << "Prediction Result: " << result << std::endl;
        std::cout << "Softmax Value: " << output << std::endl;

        cv::waitKey();
    }

    return 0;
}
