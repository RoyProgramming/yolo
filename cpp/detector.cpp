#include "detector.h"
#include <fstream>
#include <opencv2/dnn.hpp>
#include <iostream>

rsm::detGen::Detector::Detector() {}

bool rsm::detGen::Detector::init()
{
    // Инициализация настроек
    settings_.input_width = 640.0;
    settings_.input_height = 640.0;
    settings_.score_threshold = 0.5;
    settings_.nms_threshold = 0.4;
    settings_.confidence_threshold = 0.5;
    settings_.dimensions = 85;
    settings_.rows = 25200;

    colors_ = {
        cv::Scalar(0, 255, 255),
        cv::Scalar(255, 255, 0),
        cv::Scalar(255, 0, 255),
        cv::Scalar(0, 255, 0)
    };

    net_ = cv::dnn::readNet("config_files/yolov5s.onnx");
    if (net_.empty())
    {
        std::cerr << "Ошибка при загрузке модели\n";
        return false;
    }

    // Настройка на использование CPU по умолчанию
    net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    // Загрузка списка классов
    class_list_ = load_class_list();

    return true;
}

void rsm::detGen::Detector::setCuda()
{
    std::cout << "Попытка использовать CUDA\n";
    net_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
}

std::vector<rsm::type::Detection> rsm::detGen::Detector::detect(cv::Mat &image)
{
    cv::Mat blob;
    auto input_image = format_yolov5(image);

    cv::dnn::blobFromImage(input_image, blob, 1.0 / 255.0, cv::Size(settings_.input_width, settings_.input_height), cv::Scalar(), true, false);
    net_.setInput(blob);
    std::vector<cv::Mat> outputs;
    net_.forward(outputs, net_.getUnconnectedOutLayersNames());

    float x_factor = input_image.cols / settings_.input_width;
    float y_factor = input_image.rows / settings_.input_height;

    float *data = (float *)outputs[0].data;

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for (int i = 0; i < settings_.rows; ++i)
    {
        float confidence = data[4];
        if (confidence >= settings_.confidence_threshold)
        {
            float *classes_scores = data + 5;
            cv::Mat scores(1, class_list_.size(), CV_32FC1, classes_scores);
            cv::Point class_id;
            double max_class_score;
            cv::minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
            if (max_class_score > settings_.score_threshold)
            {
                confidences.push_back(confidence);
                class_ids.push_back(class_id.x);

                float x = data[0];
                float y = data[1];
                float w = data[2];
                float h = data[3];
                int left = int((x - 0.5 * w) * x_factor);
                int top = int((y - 0.5 * h) * y_factor);
                int width = int(w * x_factor);
                int height = int(h * y_factor);
                boxes.push_back(cv::Rect(left, top, width, height));
            }
        }
        data += settings_.dimensions;
    }

    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, settings_.score_threshold, settings_.nms_threshold, nms_result);

    std::vector<rsm::type::Detection> output;
    for (int idx : nms_result)
    {
        rsm::type::Detection result;
        result.class_id = class_ids[idx];
        result.confidence = confidences[idx];
        result.box = boxes[idx];
        output.push_back(result);
    }
    return output;
}

std::vector<std::string> rsm::detGen::Detector::load_class_list() const
{
    std::vector<std::string> class_list;
    std::ifstream ifs("config_files/classes.txt");
    std::string line;
    while (getline(ifs, line))
    {
        class_list.push_back(line);
    }
    return class_list;
}

cv::Mat rsm::detGen::Detector::format_yolov5(const cv::Mat &source) const
{
    int col = source.cols;
    int row = source.rows;
    int _max = std::max(col, row);
    cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
    source.copyTo(result(cv::Rect(0, 0, col, row)));
    return result;
}
