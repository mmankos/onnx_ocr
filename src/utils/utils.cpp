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
