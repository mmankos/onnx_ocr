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

#include "utils/utils.h"

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
        const std::string                                &limit_type,
        const std::shared_ptr<const std::vector<int64_t>> image_shape,
        const std::pair<const std::string, std::shared_ptr<cv::Mat>> &image);
    void preprocess();

  private:
    const bool                                        keep_ratio;
    const int                                         side_length_limit;
    const std::string                                 limit_type;
    const std::shared_ptr<const std::vector<int64_t>> image_shape;
    const std::pair<const std::string, std::shared_ptr<cv::Mat>> &image;

    int image_height;
    int image_width;
    int image_channels;

    ResizeMode resize_mode;
    float      image_resize_ratio_height;
    float      image_resize_ratio_width;

    int  resize();
    void pad_image();
    void normalize_image();

    std::pair<int, int> compute_resize_dims() const;
    bool                apply_resize(int resize_height, int resize_width);
};
