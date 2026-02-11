#pragma once

#include <algorithm>
#include <array>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <vector>

using Box = std::array<cv::Point2f, 4>;

cv::Mat             hwc_to_chw(const cv::Mat &hwc);
std::vector<size_t> sort_box_indices(const std::vector<Box> &boxes,
                                     float row_y_threshold = 10.0f);
cv::Mat             image_box_crop(const cv::Mat &img, const Box &points);
