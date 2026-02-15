#include "ocr_pipeline/recognition/preprocess/recognition_preprocess.h"

RecognitionPreprocess::RecognitionPreprocess(std::vector<int> rec_image_shape,
                                             int model_width, float scale,
                                             std::vector<float> mean,
                                             std::vector<float> std)
    : rec_image_shape(std::move(rec_image_shape)), model_width(model_width),
      scale(scale), mean(std::move(mean)), std(std::move(std))
{}

int RecognitionPreprocess::compute_target_width(float max_wh_ratio) const
{
    const int image_h = rec_image_shape[1];

    int target_w = static_cast<int>(image_h * max_wh_ratio);
    if (model_width > 0)
    {
        target_w = model_width;
    }

    return target_w;
}

cv::Mat RecognitionPreprocess::resize_and_normalize(const cv::Mat &image,
                                                    int target_width) const
{
    const int image_c = rec_image_shape[0];
    const int image_h = rec_image_shape[1];

    const int h = image.rows;
    const int w = image.cols;
    if (h <= 0 || w <= 0 || target_width <= 0)
    {
        return {};
    }

    const float ratio     = static_cast<float>(w) / h;
    int         resized_w = static_cast<int>(std::ceil(image_h * ratio));
    if (resized_w > target_width)
    {
        resized_w = target_width;
    }
    if (resized_w <= 0)
    {
        return {};
    }

    cv::Mat resized;
    cv::resize(image, resized, cv::Size(resized_w, image_h));
    resized.convertTo(resized, CV_32F);

    cv::Mat            chw        = hwc_to_chw(resized);
    std::vector<float> local_mean = mean;
    std::vector<float> local_std  = std;
    if (local_mean.size() != 3 || local_std.size() != 3)
    {
        local_mean = {0.485f, 0.456f, 0.406f};
        local_std  = {0.229f, 0.224f, 0.225f};
    }
    for (int c = 0; c < image_c; ++c)
    {
        const float m = local_mean[c];
        const float s = local_std[c];
        for (int y = 0; y < image_h; ++y)
        {
            float *row = chw.ptr<float>(c, y);
            for (int x = 0; x < resized_w; ++x)
            {
                row[x] = (row[x] * scale - m) / s;
            }
        }
    }

    int     pad_sizes[] = {image_c, image_h, target_width};
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

std::vector<size_t> RecognitionPreprocess::sort_indices_by_aspect(
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
