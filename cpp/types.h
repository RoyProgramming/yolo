#ifndef TYPES_H
#define TYPES_H

#include <opencv2/opencv.hpp>

// Настройки детекции
namespace rsm::type 
{

struct Detection
{
    int class_id;
    float confidence;
    cv::Rect box;
};

// Настройки генератора
struct Settings
{
    float input_width;
    float input_height;
    float score_threshold;
    float nms_threshold;
    float confidence_threshold;
    int dimensions;
    int rows;
};

} // namespace type

#endif // TYPES_H
