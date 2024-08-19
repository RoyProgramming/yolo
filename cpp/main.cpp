#include "detector.h"
#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <vector>
#include <iomanip>

void drawResult(cv::Mat &img, const rsm::type::Detection &detection)
{
    auto box = detection.box;
    auto classId = detection.class_id;
    const auto color = cv::Scalar(0, 255, 255);

    cv::rectangle(img, box, color, 3);

    cv::rectangle(img, cv::Point(box.x, box.y - 20), cv::Point(box.x + box.width, box.y), color, cv::FILLED);
    cv::putText(img, std::to_string(classId), cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
}

int main(int argc, char **argv)
{
    bool is_cuda = argc > 1 && strcmp(argv[1], "cuda") == 0;
    rsm::detGen::Detector detector;
    
    // Инициализация детектора без параметров
    if (!detector.init())
    {
        std::cerr << "Ошибка инициализации детектора\n";
        return -1;
    }

    // Включение CUDA, если указано в аргументах
    if (is_cuda)
    {
        detector.setCuda(); // Нужно добавить метод setCuda() для включения CUDA
    }

    cv::Mat frame;
    cv::VideoCapture capture("sample.mp4");
    if (!capture.isOpened())
    {
        std::cerr << "Ошибка открытия видеофайла\n";
        return -1;
    }

    auto start = std::chrono::high_resolution_clock::now();
    int frame_count = 0;
    float fps = -1;
    int total_frames = 0;

    while (true)
    {
        capture.read(frame);
        if (frame.empty())
        {
            std::cout << "Конец потока\n";
            break;
        }

        auto detections = detector.detect(frame);

        frame_count++;
        total_frames++;

        for (const auto &detection : detections)
        {
            drawResult(frame, detection);
        }

        if (frame_count >= 30)
        {
            auto end = std::chrono::high_resolution_clock::now();
            fps = frame_count * 1000.0 / std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

            frame_count = 0;
            start = std::chrono::high_resolution_clock::now();
        }

        if (fps > 0)
        {
            std::ostringstream fps_label;
            fps_label << std::fixed << std::setprecision(2);
            fps_label << "FPS: " << fps;
            cv::putText(frame, fps_label.str(), cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
        }

        cv::imshow("output", frame);

        if (cv::waitKey(1) != -1)
        {
            capture.release();
            std::cout << "Завершено пользователем\n";
            break;
        }
    }

    std::cout << "Всего кадров: " << total_frames << ";" << std::endl;  // Исправлено: добавлена точка с запятой

    return 0;
}
