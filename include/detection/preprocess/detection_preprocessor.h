#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <optional>
#include <string>
#include <vector>

enum class ResizeMode
{
    FixedHeightWidthVariableRatio = 0,
    FixedHeightVariableWidthRatio = 1,
    ResizeLongSide                = 2
};

class DetectionPreprocessor
{
  public:
    DetectionPreprocessor(
        const bool keep_ratio, const int side_length_limit,
        const std::string &limit_type,
        const std::optional<std::pair<const int64_t, const int64_t>> image_shape,
        std::pair<const std::string, cv::Mat> &image);
    void preprocess();

  private:
    const bool        keep_ratio;
    const int         side_length_limit;
    const std::string limit_type;
    const std::optional<std::pair<const int64_t, const int64_t>> image_shape;
    std::pair<const std::string, cv::Mat>                       &image;
    int                                                          h;
    int                                                          w;
    int                                                          c;
    ResizeMode                                                   resizeMode;

    std::optional<std::pair<cv::Mat, cv::Vec2f>> resize() const;
    cv::Mat                                      pad_image() const;
};
