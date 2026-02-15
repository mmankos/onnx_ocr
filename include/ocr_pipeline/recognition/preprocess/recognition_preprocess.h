#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <vector>

#include "utils/utils.h"

class RecognitionPreprocess
{
  public:
    RecognitionPreprocess(std::vector<int> rec_image_shape, int model_width,
                          float scale, std::vector<float> mean,
                          std::vector<float> std);

    cv::Mat resize_and_normalize(const cv::Mat &image, int target_width) const;
    int     compute_target_width(float max_wh_ratio) const;
    std::vector<size_t>
    sort_indices_by_aspect(const std::vector<cv::Mat> &images) const;

  private:
    std::vector<int>   rec_image_shape;
    int                model_width;
    float              scale;
    std::vector<float> mean;
    std::vector<float> std;
};
