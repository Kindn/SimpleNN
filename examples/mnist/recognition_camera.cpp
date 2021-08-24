#include <SimpleNN.hpp>
#include <LeNet/LeNet.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>
#include <vector>

const std::string weight_path = "../../../examples/mnist/LeNet.weights";
snn::LeNet net;

cv::Mat element = cv::getStructuringElement(0, cv::Size(3, 3));

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


void recognize(cv::Mat& _src_gray, 
               std::vector<cv::Rect>& _boxes, 
               std::vector<int>& _results, 
               double _threshold = 0.7, 
               bool _reverse = true)
{
    _boxes.clear();
    _results.clear();
    // threshold
    cv::Mat thr_img;
    if (_reverse)
        cv::threshold(_src_gray, thr_img, 50, 255, cv::THRESH_BINARY_INV);
    else
        cv::threshold(_src_gray, thr_img, 50, 255, cv::THRESH_BINARY);
    cv::dilate(thr_img, thr_img, element);
    cv::erode(thr_img, thr_img, element);
    cv::imshow("threshold", thr_img);
    // find contours
    cv::Mat hierarchy;
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(thr_img, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
    // predict
    for (int i = 0; i < contours.size(); i++)
    {
        cv::Rect rect = cv::boundingRect(contours[i]);
        cv::Mat roi = thr_img(rect);
        int pad_h = roi.rows / 4, pad_w = roi.cols / 4;
        cv::copyMakeBorder(roi, roi, pad_h, pad_h, pad_w, pad_w, cv::BORDER_CONSTANT);
        cv::resize(roi, roi, cv::Size(28, 28));

        Matrix_d input;
        cvmat2nndata(roi, input, false);
        Matrix_d output = net.predict(input);
        output = snn::softmax(output, 1);
        std::vector<int> max_cols;
        snn::argmax_axis<double>(output, max_cols, 1);
        int result = max_cols[0];
        double softmax = output(result, 0);
        if (softmax > _threshold)
        {
            _boxes.push_back(rect);
            _results.push_back(result);
        }
    }
}

int main()
{
    if (!net.load_weights(weight_path))
    {
        std::cerr << "Failed to load weights!" << std::endl;
        return -1;
    
    }
    cv::VideoCapture cap;
    cap.open(0);

    cv::Mat frame;
    while (cap.isOpened())
    {
        cap >> frame;
        if (frame.empty())
        {
            std::cout << "empty" << std::endl;
            break;
        }

        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        std::vector<cv::Rect> boxes;
        std::vector<int> results;
        recognize(gray, boxes, results, 0.9, true);
        for (int i = 0; i < boxes.size(); i++)
        {
            cv::rectangle(frame, boxes[i], cv::Scalar(0, 255, 0));
            cv::putText(frame, std::to_string(results[i]), boxes[i].tl(), 
                        cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0));
        }

        cv::imshow("camera", frame);
    
        if (cv::waitKey(5) < 255)
        {
            break;
        }
            
    }

    return 0;
}
