#include "detection/preprocess/detection_preprocessor.h"

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
    hwc_to_chw();
}

int DetectionPreprocessor::resize()
{
    int resize_height = 0;
    int resize_width  = 0;

    if (image_height + image_width < 64)
    {
        pad_image();
    }

    if (resize_mode == ResizeMode::FixedHeightWidthVariableRatio)
    { // Resize the image based on different limit types.
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
            std::cerr
                << "[ERROR][DetectionPreprocessor::resize] Unknown limit_type: "
                << limit_type << std::endl;
            return -1;
        }

        resize_height = static_cast<int>(image_height * ratio);
        resize_width  = static_cast<int>(image_width * ratio);

        resize_height =
            std::max(int(std::round(resize_height / 32.0f) * 32), 32);
        resize_width = std::max(int(std::round(resize_width / 32.0f) * 32), 32);
    }
    else if (resize_mode == ResizeMode::FixedHeightVariableWidthRatio)
    { // Resize the image while maintaining the aspect ratio.
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
    { // Resize the image based on the longer dimension.
        int long_side = std::max(image_height, image_width);

        float ratio = static_cast<float>(side_length_limit) /
                      static_cast<float>(long_side);

        resize_height = static_cast<int>(image_height * ratio);
        resize_width  = static_cast<int>(image_width * ratio);

        resize_height = ((resize_height + 127) / 128) * 128;
        resize_width  = ((resize_width + 127) / 128) * 128;
    }
    else
    {
        std::cerr << "[ERROR][DetectionPreprocessor::resize] Unknown ResizeMode"
                  << std::endl;
        return -1;
    }

    if (resize_height <= 0 || resize_width <= 0)
    {
        return -1;
    }

    cv::resize(*image.second, *image.second,
               cv::Size(resize_width, resize_height), 0, 0, cv::INTER_LINEAR);

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
    *image.second =
        cv::Mat(std::max(32, image_height), std::max(32, image_width),
                image.second->type(), cv::Scalar(0, 0, 0));
    (*image.second)(cv::Rect(0, 0, image_width, image_height)) = *image.second;
}

void DetectionPreprocessor::normalize_image()
{
    float                    scale = 1.0 / 255.0;
    const std::vector<float> mean  = {0.485f, 0.456f, 0.406f};
    const std::vector<float> std   = {0.229f, 0.224f, 0.225f};

    image.second->convertTo(*image.second, CV_32F);
    *image.second *= scale;

    for (int r = 0; r < image_height; ++r)
    {
        cv::Vec3f *row_ptr = image.second->ptr<cv::Vec3f>(r);
        for (int c = 0; c < image_width; ++c)
        {
            cv::Vec3f &pixel = row_ptr[c];
            for (int ch = 0; ch < image_channels; ++ch)
            {
                pixel[ch] = (pixel[ch] - mean[ch]) / std[ch];
            }
        }
    }
}

void DetectionPreprocessor::hwc_to_chw()
{
    int     sizes[] = {3, image_height, image_width};
    cv::Mat chw_image(3, sizes, CV_32F);

    for (int h = 0; h < image_height; ++h)
    {
        for (int w = 0; w < image_width; ++w)
        {
            cv::Vec3f pixel = image.second->at<cv::Vec3f>(h, w);

            chw_image.at<float>(0, h, w) = pixel[0];
            chw_image.at<float>(1, h, w) = pixel[1];
            chw_image.at<float>(2, h, w) = pixel[2];
        }
    }

    *image.second = chw_image;
}
