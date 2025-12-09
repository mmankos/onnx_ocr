#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <onnxruntime_cxx_api.h>
#include <opencv2/highgui.hpp>
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
        const std::string                               &limit_type,
        const std::optional<std::pair<int64_t, int64_t>> image_shape,
        const std::pair<const std::string, std::shared_ptr<cv::Mat>> &image);
    void preprocess();

  private:
    const bool                                       keep_ratio;
    const int                                        side_length_limit;
    const std::string                                limit_type;
    const std::optional<std::pair<int64_t, int64_t>> image_shape;
    const std::pair<const std::string, std::shared_ptr<cv::Mat>> &image;

    int image_height;
    int image_width;
    int image_channels;

    ResizeMode resizeMode;
    float      image_resize_ratio_height;
    float      image_resize_ratio_width;

    int  resize();
    void pad_image();
    void normalize_image();
    void hwc_to_chw();
};
