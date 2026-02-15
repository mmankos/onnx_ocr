#include "ocr_pipeline/detection/preprocess/detection_preprocess.h"

DetectionPreprocessor::DetectionPreprocessor(
    const bool keep_ratio, const int side_length_limit,
    const std::string                                            &limit_type,
    const std::shared_ptr<const std::vector<int64_t>>             image_shape,
    const std::pair<const std::string, std::shared_ptr<cv::Mat>> &image)
    : keep_ratio(keep_ratio), side_length_limit(side_length_limit),
      limit_type(limit_type), image_shape(image_shape), image(image)
{
    if (image.second->empty())
    {
        throw std::invalid_argument("[ERROR][DetectionPreprocessor::"
                                    "DetectionPreprocessor] Input image " +
                                    image.first + "is empty (null image).");
    }

    image_height   = image.second->rows;
    image_width    = image.second->cols;
    image_channels = image.second->channels();

    if (!image_shape->empty() && image_shape->size() == 2 &&
        (*image_shape)[0] && (*image_shape)[1])
    {
        resize_mode = ResizeMode::FixedHeightVariableWidthRatio;
    }
    else
    {
        resize_mode = limit_type == "resize_long"
                          ? ResizeMode::ResizeLongSide
                          : ResizeMode::FixedHeightWidthVariableRatio;
    }
}

void DetectionPreprocessor::preprocess()
{
    int resize_result = resize();
    if (resize_result == -1)
    {
        std::cerr << "[ERROR][DetectionPreprocessor::preprocess] Resize "
                     "operation failed for "
                  << image.first << std::endl;
        return;
    }
    normalize_image();
    *image.second = hwc_to_chw(*image.second);
}

int DetectionPreprocessor::resize()
{
    if (image_height + image_width < 64)
    {
        pad_image();
    }
    auto [resize_height, resize_width] = compute_resize_dims();
    if (!apply_resize(resize_height, resize_width))
    {
        return -1;
    }

    if (resize_mode == ResizeMode::FixedHeightWidthVariableRatio)
    {
        image_resize_ratio_height =
            static_cast<float>(resize_height) / image_height;
        image_resize_ratio_width =
            static_cast<float>(resize_width) / image_width;
    }
    else if (resize_mode == ResizeMode::FixedHeightVariableWidthRatio)
    {
        image_resize_ratio_height = static_cast<float>(resize_height) /
                                    static_cast<float>(image_height);
        image_resize_ratio_width =
            static_cast<float>(resize_width) / static_cast<float>(image_width);
    }
    else if (resize_mode == ResizeMode::ResizeLongSide)
    {
        image_resize_ratio_height =
            static_cast<float>(resize_height) / image_height;
        image_resize_ratio_width =
            static_cast<float>(resize_width) / image_width;
    }
    else
    {
        std::cerr << "[ERROR][DetectionPreprocessor::resize] Unknown ResizeMode"
                  << std::endl;
        return -1;
    }

    image_height = image.second->rows;
    image_width  = image.second->cols;
    return 0;
}

void DetectionPreprocessor::pad_image()
{
    cv::Mat old_image = *image.second;
    cv::Mat padded_image(std::max(32, image_height), std::max(32, image_width),
                         old_image.type(), cv::Scalar(0, 0, 0));

    old_image.copyTo(padded_image(cv::Rect(0, 0, image_width, image_height)));

    *image.second = padded_image;
}

void DetectionPreprocessor::normalize_image()
{
    const cv::Scalar mean(0.485f, 0.456f, 0.406f);
    const cv::Scalar std_dev(0.229f, 0.224f, 0.225f);

    image.second->convertTo(*image.second, CV_32F, 1.0 / 255.0);

    std::vector<cv::Mat> channels;
    cv::split(*image.second, channels);
    for (size_t i = 0; i < channels.size(); ++i)
    {
        channels[i] = (channels[i] - mean[i]) / std_dev[i];
    }
    cv::merge(channels, *image.second);
}

std::pair<int, int> DetectionPreprocessor::compute_resize_dims() const
{
    int resize_height = 0;
    int resize_width  = 0;

    if (resize_mode == ResizeMode::FixedHeightWidthVariableRatio)
    {
        float ratio = 1.0f;

        if (limit_type == "max" &&
            std::max(image_height, image_width) > side_length_limit)
        {
            ratio = static_cast<float>(side_length_limit) /
                    std::max(image_height, image_width);
        }
        else if (limit_type == "min" &&
                 std::min(image_height, image_width) < side_length_limit)
        {
            ratio = static_cast<float>(side_length_limit) /
                    std::min(image_height, image_width);
        }
        else if (limit_type == "resize_long")
        {
            ratio = static_cast<float>(side_length_limit) /
                    std::max(image_height, image_width);
        }
        else
        {
            return {0, 0};
        }

        resize_height = static_cast<int>(image_height * ratio);
        resize_width  = static_cast<int>(image_width * ratio);

        resize_height =
            std::max(int(std::round(resize_height / 32.0f) * 32), 32);
        resize_width = std::max(int(std::round(resize_width / 32.0f) * 32), 32);
    }
    else if (resize_mode == ResizeMode::FixedHeightVariableWidthRatio)
    {
        resize_height = (*image_shape)[0];
        resize_width  = (*image_shape)[1];

        if (keep_ratio)
        {
            float w_scaled = (static_cast<float>(image_width) * resize_height) /
                             static_cast<float>(image_height);
            resize_width = static_cast<int>(std::ceil(w_scaled / 32.0)) * 32;
        }
    }
    else if (resize_mode == ResizeMode::ResizeLongSide)
    {
        int long_side = std::max(image_height, image_width);

        float ratio = static_cast<float>(side_length_limit) /
                      static_cast<float>(long_side);

        resize_height = static_cast<int>(image_height * ratio);
        resize_width  = static_cast<int>(image_width * ratio);

        resize_height = ((resize_height + 127) / 128) * 128;
        resize_width  = ((resize_width + 127) / 128) * 128;
    }

    return {resize_height, resize_width};
}

bool DetectionPreprocessor::apply_resize(int resize_height, int resize_width)
{
    if (resize_height <= 0 || resize_width <= 0)
    {
        std::cerr
            << "[ERROR][DetectionPreprocessor::resize] Unknown limit_type: "
            << limit_type << std::endl;
        return false;
    }

    cv::resize(*image.second, *image.second,
               cv::Size(resize_width, resize_height), 0, 0, cv::INTER_LINEAR);
    return true;
}
