#ifndef DETECTOR_H
#define DETECTOR_H

#include <opencv2/opencv.hpp>
#include <vector>
#include "types.h"

namespace rsm::detGen 
{

    class Detector
    {
    public:
        Detector();

        std::vector<rsm::type::Detection> detect(cv::Mat &image);

        bool init();
        void setCuda();  // Добавляем метод для включения CUDA

    private:
        rsm::type::Settings settings_;
        cv::dnn::Net net_;
        std::vector<std::string> class_list_;
        std::vector<cv::Scalar> colors_;

        std::vector<std::string> load_class_list() const;
        cv::Mat format_yolov5(const cv::Mat &source) const;
    };

} // namespace rsm

#endif // DETECTOR_H
