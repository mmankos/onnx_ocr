#include "detection/preprocess/detection_preprocessor.h"

DetectionPreprocessor::DetectionPreprocessor(
    const bool keep_ratio, const int side_length_limit,
    const std::string                                           &limit_type,
    const std::optional<std::pair<const int64_t, const int64_t>> image_shape,
    std::pair<const std::string, cv::Mat>                       &image)
    : keep_ratio(keep_ratio), side_length_limit(side_length_limit),
      limit_type(limit_type), image_shape(image_shape), image(image)
{
    if (image.second.empty())
    {
        throw std::invalid_argument("[ERROR][DetectionPreprocessor::"
                                    "DetectionPreprocessor] Input image " +
                                    image.first + "is empty (null image).");
    }

    h = image.second.rows;
    w = image.second.cols;
    c = image.second.channels();

    if (image_shape)
    {
        resizeMode = ResizeMode::FixedHeightVariableWidthRatio;
    }
    else
    {
        resizeMode = limit_type == "resize_long"
                         ? ResizeMode::ResizeLongSide
                         : ResizeMode::FixedHeightWidthVariableRatio;
    }
}

void DetectionPreprocessor::preprocess()
{
    std::optional<std::pair<cv::Mat, cv::Vec2f>> resize_result = resize();
    if (!resize_result)
    {
        std::cerr << "[ERROR][DetectionPreprocessor::preprocess] Resize "
                     "operation failed for "
                  << image.first << std::endl;
        return;
    }
}

std::optional<std::pair<cv::Mat, cv::Vec2f>>
DetectionPreprocessor::resize() const
{
    int resize_h = 0;
    int resize_w = 0;

    if (h + w < 64)
    {
        image.second = pad_image();
    }

    if (resizeMode == ResizeMode::FixedHeightWidthVariableRatio)
    { // Resize the image based on different limit types.
        float ratio = 1.0f;

        if (limit_type == "max" && std::max(h, w) > side_length_limit)
        {
            ratio = static_cast<float>(side_length_limit) / std::max(h, w);
        }
        else if (limit_type == "min" && std::min(h, w) < side_length_limit)
        {
            ratio = static_cast<float>(side_length_limit) / std::min(h, w);
        }
        else if (limit_type == "resize_long")
        {
            ratio = static_cast<float>(side_length_limit) / std::max(h, w);
        }
        else
        {
            std::cerr << "[ERROR] Unknown limit_type: " << limit_type
                      << std::endl;
            return std::nullopt;
        }

        resize_h = static_cast<int>(h * ratio);
        resize_w = static_cast<int>(w * ratio);

        resize_h = std::max(int(std::round(resize_h / 32.0f) * 32), 32);
        resize_w = std::max(int(std::round(resize_w / 32.0f) * 32), 32);
    }
    else if (resizeMode == ResizeMode::FixedHeightVariableWidthRatio)
    { // Resize the image while maintaining the aspect ratio.
        resize_h = image_shape->first;
        resize_w = image_shape->second;

        if (keep_ratio)
        {
            float w_scaled =
                (static_cast<float>(w) * resize_h) / static_cast<float>(h);
            resize_w = static_cast<int>(std::ceil(w_scaled / 32.0)) * 32;
        }
    }
    else if (resizeMode == ResizeMode::ResizeLongSide)
    { // Resize the image based on the longer dimension.
        int long_side = std::max(h, w);

        float ratio = static_cast<float>(side_length_limit) /
                      static_cast<float>(long_side);

        resize_h = static_cast<int>(h * ratio);
        resize_w = static_cast<int>(w * ratio);

        resize_h = ((resize_h + 127) / 128) * 128;
        resize_w = ((resize_w + 127) / 128) * 128;
    }
    else
    {
        std::cerr << "[ERROR] Unknown ResizeMode" << std::endl;
        return std::nullopt;
    }

    if (resize_h <= 0 || resize_w <= 0)
    {
        return std::nullopt;
    }

    float   ratio_h;
    float   ratio_w;
    cv::Mat resized;
    cv::resize(image.second, resized, cv::Size(resize_w, resize_h), 0, 0,
               cv::INTER_LINEAR);

    if (resizeMode == ResizeMode::FixedHeightWidthVariableRatio)
    {
        ratio_h = static_cast<float>(resize_h) / h;
        ratio_w = static_cast<float>(resize_w) / w;
    }
    else if (resizeMode == ResizeMode::FixedHeightVariableWidthRatio)
    {
        ratio_h = static_cast<float>(resize_h) / static_cast<float>(h);
        ratio_w = static_cast<float>(resize_w) / static_cast<float>(w);
    }
    else if (resizeMode == ResizeMode::ResizeLongSide)
    {
        ratio_h = static_cast<float>(resize_h) / h;
        ratio_w = static_cast<float>(resize_w) / w;
    }
    else
    {
        std::cerr << "[ERROR] Unknown ResizeMode" << std::endl;
        return std::nullopt;
    }

    return std::make_pair(resized, cv::Vec2f(ratio_h, ratio_w));
}

cv::Mat DetectionPreprocessor::pad_image() const
{
    int new_h = std::max(32, h);
    int new_w = std::max(32, w);

    cv::Mat img_pad(new_h, new_w, image.second.type(), cv::Scalar(0, 0, 0));

    image.second.copyTo(img_pad(cv::Rect(0, 0, w, h)));

    return img_pad;
}
