#pragma once

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <opencv2/opencv.hpp>
#include <string>
#include <utility>
#include <vector>

#include "utils/utils.h"

class DetectionPostprocessor
{
  public:
    struct BoxResult
    {
        std::vector<cv::Point2f> points;
        float                    score;
    };

    DetectionPostprocessor(float threshold = 0.3f, float box_threshold = 0.7f,
                           int max_candidates = 1000, float unclip_ratio = 2.0f,
                           bool use_dilation = false);

    std::vector<Box> postprocess(const cv::Mat &prediction_maps,
                                 const int64_t  original_image_height,
                                 const int64_t  original_image_width) const;

  private:
    float   threshold;
    float   box_threshold;
    int     max_candidates;
    float   unclip_ratio;
    int     min_size;
    cv::Mat dilation_kernel;

    std::vector<BoxResult> boxes_from_bitmap(const cv::Mat &pred,
                                             const cv::Mat &bitmap,
                                             int            dest_width,
                                             int            dest_height) const;

    cv::Mat extract_prediction_map(const cv::Mat &prediction_maps) const;
    cv::Mat build_segmentation_mask(const cv::Mat &pred_map) const;
    std::vector<Box>
    convert_boxes(const std::vector<BoxResult> &boxes_result) const;

    std::vector<std::vector<cv::Point2f>>
    unclip(const std::vector<cv::Point2f> &box, float unclip_ratio) const;

    std::pair<std::vector<cv::Point2f>, float>
    get_mini_boxes(const std::vector<cv::Point2f> &contour) const;

    float box_score_slow(const cv::Mat                  &bitmap,
                         const std::vector<cv::Point2f> &contour) const;

    Box order_points_clockwise(const Box &points) const;

    Box clip_points_to_image(const Box &points, int image_height,
                             int image_width) const;

    std::vector<Box> filter_small_and_clip_boxes(const std::vector<Box> &boxes,
                                                 int image_height,
                                                 int image_width) const;
};
