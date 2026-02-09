#include "detection/detection.h"

Detector::Detector(
    Ort::Env &env, const Ort::SessionOptions &session_options,
    Ort::MemoryInfo &memory_info, const std::string &det_filepath,
    const std::unordered_map<std::string, OnnxModelInputInfo> &models,
    bool keep_ratio, int side_length_limit, const std::string &limit_type)
    : env(env), session_options(session_options), memory_info(memory_info),
      det_filepath(det_filepath), models(models), keep_ratio(keep_ratio),
      side_length_limit(side_length_limit), limit_type(limit_type)
{
    postprocessor = std::make_unique<DetectionPostprocessor>();
}

const std::string &Detector::get_limit_type() const { return limit_type; }

void Detector::set_limit_type(std::string limit_type)
{
    this->limit_type = std::move(limit_type);
}

std::vector<std::array<cv::Point2f, 4>> Detector::run(
    const std::pair<const std::string, std::shared_ptr<cv::Mat>> &image)
{
    const int64_t original_image_height = image.second->rows;
    const int64_t original_image_width  = image.second->cols;

    auto model_it = models.find(det_filepath);
    if (model_it == models.end())
    {
        return {};
    }

    std::unique_ptr<DetectionPreprocessor> detection_preprocessor =
        std::make_unique<DetectionPreprocessor>(
            keep_ratio, side_length_limit, limit_type,
            model_it->second.image_shape, image);

    detection_preprocessor->preprocess();

    int64_t image_channels = image.second->channels();
    int64_t image_height   = image.second->rows;
    int64_t image_width    = image.second->cols;

    if (image.second->dims == 3)
    {
        image_channels = image.second->size[0];
        image_height   = image.second->size[1];
        image_width    = image.second->size[2];
    }

    if (model_it->second.image_shape->empty() ||
        model_it->second.image_shape->size() != 2)
    {
        model_it->second.image_shape->resize(2);
        (*model_it->second.image_shape)[0] = image_height;
        (*model_it->second.image_shape)[1] = image_width;
    }

    std::vector<int64_t>        input_shape = {1, image_channels, image_height,
                                               image_width};
    const std::vector<int64_t> &model_shape = model_it->second.shape;

    if (model_shape.size() == 4)
    {
        input_shape = model_shape;
        if (input_shape[0] <= 0)
            input_shape[0] = 1;
        if (input_shape[1] <= 0)
            input_shape[1] = image_channels;
        if (input_shape[2] <= 0)
            input_shape[2] = image_height;
        if (input_shape[3] <= 0)
            input_shape[3] = image_width;
    }
    else if (model_shape.size() == 3)
    {
        input_shape = model_shape;
        if (input_shape[0] <= 0)
            input_shape[0] = image_channels;
        if (input_shape[1] <= 0)
            input_shape[1] = image_height;
        if (input_shape[2] <= 0)
            input_shape[2] = image_width;
    }
    else if (model_shape.size() == 2)
    {
        input_shape = model_shape;
        if (input_shape[0] <= 0)
            input_shape[0] = image_height;
        if (input_shape[1] <= 0)
            input_shape[1] = image_width;
    }

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, reinterpret_cast<float *>(image.second->data),
        image.second->total(), input_shape.data(), input_shape.size());

    Ort::AllocatorWithDefaultOptions allocator;
    std::vector<const char *> input_names = {model_it->second.name.c_str()};

    std::unique_ptr<Ort::Session> session = std::make_unique<Ort::Session>(
        env, det_filepath.c_str(), session_options);

    size_t                    output_count = session->GetOutputCount();
    std::vector<const char *> output_names;
    std::vector<Ort::AllocatedStringPtr> output_name_allocs;
    output_names.reserve(output_count);
    output_name_allocs.reserve(output_count);

    for (size_t i = 0; i < output_count; ++i)
    {
        auto output_name = session->GetOutputNameAllocated(i, allocator);
        output_names.push_back(output_name.get());
        output_name_allocs.push_back(std::move(output_name));
    }

    std::vector<Ort::Value> output_tensors = session->Run(
        Ort::RunOptions{nullptr}, input_names.data(), &input_tensor, 1,
        output_names.data(), output_names.size());

    Ort::Value &output_tensor = output_tensors[0];
    auto        output_info   = output_tensor.GetTensorTypeAndShapeInfo();
    auto        output_shape  = output_info.GetShape();

    if (output_shape.size() != 4)
    {
        return {};
    }

    int sizes[] = {
        static_cast<int>(output_shape[0]), static_cast<int>(output_shape[1]),
        static_cast<int>(output_shape[2]), static_cast<int>(output_shape[3])};

    cv::Mat pred_maps(4, sizes, CV_32F,
                      output_tensor.GetTensorMutableData<float>());

    return postprocessor->postprocess(pred_maps, original_image_height,
                                      original_image_width);
}
