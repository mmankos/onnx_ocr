#include "ocr_pipeline/detection/detection.h"

Detector::Detector(Ort::Env &env, const Ort::SessionOptions &session_options,
                   Ort::MemoryInfo   &memory_info,
                   const std::string &model_filepath,
                   std::unordered_map<std::string, OnnxModelInputInfo> &models,
                   bool keep_ratio, int side_length_limit,
                   const std::string &limit_type, float threshold,
                   float box_threshold, int max_candidates, float unclip_ratio,
                   bool use_dilation)
    : env(env), session_options(session_options), memory_info(memory_info),
      model_filepath(model_filepath), models(models), keep_ratio(keep_ratio),
      side_length_limit(side_length_limit), limit_type(limit_type)
{
    postprocessor = std::make_unique<DetectionPostprocessor>(
        threshold, box_threshold, max_candidates, unclip_ratio, use_dilation);
}

const std::string &Detector::get_limit_type() const { return limit_type; }

void Detector::set_limit_type(std::string limit_type)
{
    this->limit_type = std::move(limit_type);
}

std::vector<Box> Detector::run(
    const std::pair<const std::string, std::shared_ptr<cv::Mat>> &image)
{
    const int64_t original_image_height = image.second->rows;
    const int64_t original_image_width  = image.second->cols;

    OnnxModelInputInfo *model_info = get_model_info();
    if (!model_info)
    {
        return {};
    }

    // Clear image_shape so each image gets fresh resize computation.
    // The model has dynamic dims; image_shape is only needed transiently
    // for building the input tensor shape.
    model_info->image_shape->clear();

    std::unique_ptr<DetectionPreprocessor> detection_preprocessor =
        create_preprocessor(*model_info, image);
    detection_preprocessor->preprocess();

    ImageDimensions dimensions = extract_image_dims(*image.second);
    update_image_shape(*model_info, dimensions.height, dimensions.width);
    Ort::Value input_tensor = prepare_input_tensor(*image.second, *model_info);

    std::unique_ptr<Ort::Session>        session = create_session();
    std::vector<Ort::AllocatedStringPtr> output_name_allocs;
    std::vector<const char *>            output_names =
        get_output_names(*session, output_name_allocs);

    std::vector<const char *> input_names = {model_info->name.c_str()};
    std::vector<Ort::Value>   output_tensors =
        run_inference(*session, input_names, input_tensor, output_names);

    if (output_tensors.empty())
    {
        return {};
    }

    cv::Mat prediction_maps = build_prediction_maps(output_tensors[0]);
    if (prediction_maps.empty())
    {
        return {};
    }

    return postprocessor->postprocess(prediction_maps, original_image_height,
                                      original_image_width);
}

OnnxModelInputInfo *Detector::get_model_info() const
{
    auto model_it = models.find(model_filepath);
    if (model_it == models.end())
    {
        return nullptr;
    }
    return &model_it->second;
}

std::unique_ptr<DetectionPreprocessor> Detector::create_preprocessor(
    const OnnxModelInputInfo                                     &model_info,
    const std::pair<const std::string, std::shared_ptr<cv::Mat>> &image) const
{
    return std::make_unique<DetectionPreprocessor>(
        keep_ratio, side_length_limit, limit_type, model_info.image_shape,
        image);
}

Detector::ImageDimensions
Detector::extract_image_dims(const cv::Mat &image) const
{
    ImageDimensions dimensions{image.channels(), image.rows, image.cols};

    if (image.dims == 3)
    {
        dimensions.channels = image.size[0];
        dimensions.height   = image.size[1];
        dimensions.width    = image.size[2];
    }

    return dimensions;
}

void Detector::update_image_shape(OnnxModelInputInfo &model_info,
                                  int64_t             image_height,
                                  int64_t             image_width) const
{
    if (model_info.image_shape->empty() || model_info.image_shape->size() != 2)
    {
        model_info.image_shape->resize(2);
        (*model_info.image_shape)[0] = image_height;
        (*model_info.image_shape)[1] = image_width;
    }
}

std::vector<int64_t> Detector::build_input_shape(
    const OnnxModelInputInfo &model_info, int64_t image_channels,
    int64_t image_height, int64_t image_width) const
{
    std::vector<int64_t>        input_shape = {1, image_channels, image_height,
                                               image_width};
    const std::vector<int64_t> &model_shape = model_info.shape;

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

    return input_shape;
}

Ort::Value Detector::create_input_tensor(
    const cv::Mat &image, const std::vector<int64_t> &input_shape) const
{
    return Ort::Value::CreateTensor<float>(
        memory_info, reinterpret_cast<float *>(image.data), image.total(),
        input_shape.data(), input_shape.size());
}

Ort::Value Detector::prepare_input_tensor(
    const cv::Mat &image, const OnnxModelInputInfo &model_info) const
{
    ImageDimensions      dimensions  = extract_image_dims(image);
    std::vector<int64_t> input_shape = build_input_shape(
        model_info, dimensions.channels, dimensions.height, dimensions.width);
    return create_input_tensor(image, input_shape);
}

std::unique_ptr<Ort::Session> Detector::create_session() const
{
    return std::make_unique<Ort::Session>(env, model_filepath.c_str(),
                                          session_options);
}

std::vector<const char *> Detector::get_output_names(
    Ort::Session                         &session,
    std::vector<Ort::AllocatedStringPtr> &output_name_allocs) const
{
    Ort::AllocatorWithDefaultOptions allocator;
    size_t                           output_count = session.GetOutputCount();
    std::vector<const char *>        output_names;
    output_names.reserve(output_count);
    output_name_allocs.reserve(output_count);

    for (size_t i = 0; i < output_count; ++i)
    {
        auto output_name = session.GetOutputNameAllocated(i, allocator);
        output_names.push_back(output_name.get());
        output_name_allocs.push_back(std::move(output_name));
    }

    return output_names;
}

std::vector<Ort::Value> Detector::run_inference(
    Ort::Session &session, const std::vector<const char *> &input_names,
    Ort::Value                      &input_tensor,
    const std::vector<const char *> &output_names) const
{
    return session.Run(Ort::RunOptions{nullptr}, input_names.data(),
                       &input_tensor, 1, output_names.data(),
                       output_names.size());
}

cv::Mat Detector::build_prediction_maps(Ort::Value &output_tensor) const
{
    auto output_info  = output_tensor.GetTensorTypeAndShapeInfo();
    auto output_shape = output_info.GetShape();

    if (output_shape.size() != 4)
    {
        return {};
    }

    int sizes[] = {
        static_cast<int>(output_shape[0]), static_cast<int>(output_shape[1]),
        static_cast<int>(output_shape[2]), static_cast<int>(output_shape[3])};

    return cv::Mat(4, sizes, CV_32F,
                   output_tensor.GetTensorMutableData<float>());
}
