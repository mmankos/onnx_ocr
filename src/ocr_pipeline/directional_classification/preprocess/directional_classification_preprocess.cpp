#include "ocr_pipeline/directional_classification/preprocess/directional_classification_preprocess.h"

DirectionalClassificationPreprocess::DirectionalClassificationPreprocess(
    std::vector<int> cls_image_shape)
    : cls_image_shape(std::move(cls_image_shape))
{}

cv::Mat DirectionalClassificationPreprocess::resize_and_normalize(
    const cv::Mat &image) const
{
    const int image_c = cls_image_shape[0];
    const int image_h = cls_image_shape[1];
    const int image_w = cls_image_shape[2];

    const int h = image.rows;
    const int w = image.cols;
    if (h <= 0 || w <= 0)
    {
        return {};
    }

    const float ratio     = static_cast<float>(w) / h;
    int         resized_w = static_cast<int>(std::ceil(image_h * ratio));
    if (resized_w > image_w)
    {
        resized_w = image_w;
    }
    if (resized_w <= 0)
    {
        return {};
    }

    cv::Mat resized;
    cv::resize(image, resized, cv::Size(resized_w, image_h));
    resized.convertTo(resized, CV_32F, 1.0 / 255.0);

    cv::Mat chw;
    if (image_c == 1)
    {
        if (resized.channels() == 3)
        {
            cv::cvtColor(resized, resized, cv::COLOR_BGR2GRAY);
        }
        int sizes[] = {1, image_h, resized_w};
        chw         = cv::Mat(3, sizes, CV_32F);
        for (int y = 0; y < image_h; ++y)
        {
            const float *row = resized.ptr<float>(y);
            for (int x = 0; x < resized_w; ++x)
            {
                chw.at<float>(0, y, x) = row[x];
            }
        }
    }
    else
    {
        std::vector<cv::Mat> channels;
        cv::split(resized, channels);
        int sizes[] = {image_c, image_h, resized_w};
        chw         = cv::Mat(3, sizes, CV_32F);
        for (int c = 0; c < image_c; ++c)
        {
            for (int y = 0; y < image_h; ++y)
            {
                const float *row = channels[c].ptr<float>(y);
                for (int x = 0; x < resized_w; ++x)
                {
                    chw.at<float>(c, y, x) = row[x];
                }
            }
        }
    }

    chw = (chw - 0.5f) / 0.5f;

    int     pad_sizes[] = {image_c, image_h, image_w};
    cv::Mat padded(3, pad_sizes, CV_32F, cv::Scalar(0));
    for (int c = 0; c < image_c; ++c)
    {
        for (int y = 0; y < image_h; ++y)
        {
            const float *src = chw.ptr<float>(c, y);
            float       *dst = padded.ptr<float>(c, y);
            std::memcpy(dst, src, sizeof(float) * resized_w);
        }
    }

    return padded;
}

std::vector<size_t> DirectionalClassificationPreprocess::sort_indices_by_aspect(
    const std::vector<cv::Mat> &images) const
{
    std::vector<float> ratios;
    ratios.reserve(images.size());
    for (const auto &image : images)
    {
        ratios.push_back(image.cols / static_cast<float>(image.rows));
    }

    std::vector<size_t> idx(images.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::stable_sort(idx.begin(), idx.end(),
                     [&](size_t a, size_t b) { return ratios[a] < ratios[b]; });

    return idx;
}
