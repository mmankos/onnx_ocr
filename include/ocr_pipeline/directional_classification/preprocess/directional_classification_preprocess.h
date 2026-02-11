#pragma once

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <vector>

class DirectionalClassificationPreprocess
{
  public:
    DirectionalClassificationPreprocess(std::vector<int> cls_image_shape);

    cv::Mat resize_and_normalize(const cv::Mat &image) const;
    std::vector<size_t>
    sort_indices_by_aspect(const std::vector<cv::Mat> &images) const;

  private:
    std::vector<int> cls_image_shape;
};
