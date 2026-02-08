#pragma once

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/opencv.hpp>

class DetectionPostprocessor
{
  public:
    struct BoxResult
    {
        std::vector<cv::Point2f> points;
        float                    score;
    };

    DetectionPostprocessor(float threshold = 0.3f, float box_thresh = 0.7f,
                           int max_candidates = 1000, float unclip_ratio = 2.0f,
                           bool        use_dilation = false,
                           std::string score_mode   = "slow");

    std::vector<std::array<cv::Point2f, 4>>
    postprocess(const cv::Mat &pred_maps, const int64_t original_image_height,
                const int64_t original_image_width) const;

  private:
    float       threshold;
    float       box_thresh;
    int         max_candidates;
    float       unclip_ratio;
    int         min_size;
    std::string score_mode;
    cv::Mat     dilation_kernel;

    std::vector<BoxResult> boxes_from_bitmap(const cv::Mat &pred,
                                             const cv::Mat &bitmap,
                                             int            dest_width,
                                             int            dest_height) const;

    std::vector<std::vector<cv::Point2f>>
    unclip(const std::vector<cv::Point2f> &box, float unclip_ratio) const;

    std::pair<std::vector<cv::Point2f>, float>
    get_mini_boxes(const std::vector<cv::Point2f> &contour) const;

    float box_score_slow(const cv::Mat                  &bitmap,
                         const std::vector<cv::Point2f> &contour) const;

    std::array<cv::Point2f, 4>
    order_points_clockwise(const std::array<cv::Point2f, 4> &pts) const;

    std::array<cv::Point2f, 4>
    clip_points_to_image(const std::array<cv::Point2f, 4> &points,
                         int img_height, int img_width) const;

    std::vector<std::array<cv::Point2f, 4>> filter_small_and_clip_boxes(
        const std::vector<std::array<cv::Point2f, 4>> &boxes, int img_height,
        int img_width) const;
};
