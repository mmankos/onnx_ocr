#include "utils/utils.h"

cv::Mat hwc_to_chw(const cv::Mat &hwc)
{
    CV_Assert(hwc.channels() == 1 || hwc.channels() == 3);

    const int h = hwc.rows;
    const int w = hwc.cols;
    const int c = hwc.channels();

    cv::Mat hwc_f;
    if (hwc.type() != CV_32F && hwc.type() != CV_32FC1 &&
        hwc.type() != CV_32FC3)
        hwc.convertTo(hwc_f, CV_32F);
    else
        hwc_f = hwc;

    int     sizes[] = {c, h, w};
    cv::Mat chw(3, sizes, CV_32F);

    if (c == 1)
    {
        for (int y = 0; y < h; ++y)
        {
            const float *row = hwc_f.ptr<float>(y);
            for (int x = 0; x < w; ++x) { chw.at<float>(0, y, x) = row[x]; }
        }
    }
    else
    {
        for (int y = 0; y < h; ++y)
        {
            const cv::Vec3f *row = hwc_f.ptr<cv::Vec3f>(y);
            for (int x = 0; x < w; ++x)
            {
                const cv::Vec3f &pix   = row[x];
                chw.at<float>(0, y, x) = pix[0];
                chw.at<float>(1, y, x) = pix[1];
                chw.at<float>(2, y, x) = pix[2];
            }
        }
    }

    return chw;
}

std::vector<size_t> sort_box_indices(const std::vector<Box> &boxes,
                                     float                   row_y_threshold)
{
    std::vector<size_t> idx(boxes.size());
    std::iota(idx.begin(), idx.end(), 0);

    std::stable_sort(idx.begin(), idx.end(), [&](size_t a, size_t b) {
        const auto &A = boxes[a];
        const auto &B = boxes[b];
        if (A[0].y != B[0].y)
            return A[0].y < B[0].y;
        return A[0].x < B[0].x;
    });

    size_t i = 0;
    while (i < idx.size())
    {
        const float row_y = boxes[idx[i]][0].y;
        size_t      j     = i + 1;
        while (j < idx.size() &&
               std::abs(boxes[idx[j]][0].y - row_y) < row_y_threshold)
        {
            ++j;
        }

        std::stable_sort(
            idx.begin() + i, idx.begin() + j,
            [&](size_t a, size_t b) { return boxes[a][0].x < boxes[b][0].x; });

        i = j;
    }

    return idx;
}

cv::Mat image_box_crop(const cv::Mat &img, const Box &points)
{
    float w1 = cv::norm(points[0] - points[1]);
    float w2 = cv::norm(points[2] - points[3]);
    float h1 = cv::norm(points[0] - points[3]);
    float h2 = cv::norm(points[1] - points[2]);

    int crop_w = static_cast<int>(std::round(std::max(w1, w2)));
    int crop_h = static_cast<int>(std::round(std::max(h1, h2)));

    if (crop_w <= 0 || crop_h <= 0)
        return cv::Mat();

    Box targer_quad = {
        cv::Point2f(0.f, 0.f), cv::Point2f(static_cast<float>(crop_w), 0.f),
        cv::Point2f(static_cast<float>(crop_w), static_cast<float>(crop_h)),
        cv::Point2f(0.f, static_cast<float>(crop_h))};

    cv::Mat M = cv::getPerspectiveTransform(points.data(), targer_quad.data());
    cv::Mat dst;
    cv::warpPerspective(img, dst, M, cv::Size(crop_w, crop_h), cv::INTER_CUBIC,
                        cv::BORDER_REPLICATE);

    if (dst.rows > 0 && dst.cols > 0 &&
        static_cast<float>(dst.rows) / dst.cols >= 1.5f)
    {
        cv::rotate(dst, dst, cv::ROTATE_90_COUNTERCLOCKWISE);
    }

    return dst;
}
