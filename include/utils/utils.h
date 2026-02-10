#pragma once

#include <algorithm>
#include <array>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <vector>

using Box = std::array<cv::Point2f, 4>;

cv::Mat hwc_to_chw(const cv::Mat &hwc);
