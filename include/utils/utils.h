#pragma once

#include <algorithm>
#include <array>
#include <numeric>
#include <opencv2/freetype.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <unordered_map>
#include <vector>

using Box = std::array<cv::Point2f, 4>;

cv::Mat             hwc_to_chw(const cv::Mat &hwc);
std::vector<size_t> sort_box_indices(const std::vector<Box> &boxes,
                                     float row_y_threshold = 10.0f);
cv::Mat             image_box_crop(const cv::Mat &image, const Box &points);

void show_boxes(
    const cv::Mat &image, const std::vector<Box> &boxes,
    const std::string                                &window_name,
    const std::vector<size_t>                        &sorted_box_indices = {},
    const std::vector<std::pair<std::string, float>> &cls_results        = {},
    float                                             cls_threshold      = 0.0f,
    const std::unordered_map<size_t, std::string>    &rec_texts          = {});
