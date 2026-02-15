#include "ocr_pipeline/directional_classification/directional_classification.h"

DirectionalClassifier::DirectionalClassifier(
    Ort::Env &env, const Ort::SessionOptions &session_options,
    Ort::MemoryInfo &memory_info, const std::string &model_filepath,
    std::unordered_map<std::string, OnnxModelInputInfo> &models,
    std::vector<int> cls_image_shape, int cls_batch_size, float cls_threshold,
    std::vector<std::string> label_list)
    : env(env), session_options(session_options), memory_info(memory_info),
      model_filepath(model_filepath), models(models),
      cls_image_shape(std::move(cls_image_shape)),
      cls_batch_size(cls_batch_size), cls_threshold(cls_threshold)
{
    preprocessor = std::make_unique<DirectionalClassificationPreprocess>(
        this->cls_image_shape);
    postprocessor = std::make_unique<DirectionalClassificationPostprocess>(
        std::move(label_list));
}

std::pair<std::vector<cv::Mat>, std::vector<std::pair<std::string, float>>>
DirectionalClassifier::run(const std::vector<cv::Mat> &text_crops)
{
    std::vector<cv::Mat> images    = text_crops;
    const size_t         image_num = images.size();
    if (image_num == 0)
    {
        return {images, {}};
    }

    std::vector<size_t> indices = preprocessor->sort_indices_by_aspect(images);
    std::vector<std::pair<std::string, float>> cls_res(image_num, {"", 0.0f});

    std::unique_ptr<Ort::Session>        session = create_session();
    std::vector<Ort::AllocatedStringPtr> output_name_allocs;
    std::vector<const char *>            output_names =
        get_output_names(*session, output_name_allocs);

    for (size_t beg = 0; beg < image_num; beg += cls_batch_size)
    {
        size_t end =
            std::min(image_num, beg + static_cast<size_t>(cls_batch_size));
        std::vector<cv::Mat> batch;
        batch.reserve(end - beg);

        for (size_t i = beg; i < end; ++i)
        {
            cv::Mat norm_image =
                preprocessor->resize_and_normalize(images[indices[i]]);
            batch.push_back(norm_image);
        }

        const int batch_size = static_cast<int>(batch.size());
        const int c          = cls_image_shape[0];
        const int h          = cls_image_shape[1];
        const int w          = cls_image_shape[2];

        std::vector<float> batch_data(batch_size * c * h * w);
        for (int i = 0; i < batch_size; ++i)
        {
            std::memcpy(batch_data.data() + i * c * h * w,
                        batch[i].ptr<float>(0), sizeof(float) * c * h * w);
        }

        std::vector<int64_t> input_shape  = {batch_size, c, h, w};
        Ort::Value           input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, batch_data.data(), batch_data.size(),
            input_shape.data(), input_shape.size());

        std::vector<const char *> input_names;
        auto                      model_it = models.find(model_filepath);
        if (model_it == models.end())
        {
            return {images, cls_res};
        }
        input_names.push_back(model_it->second.name.c_str());

        std::vector<Ort::Value> output_tensors =
            run_inference(*session, input_names, input_tensor, output_names);
        if (output_tensors.empty())
        {
            continue;
        }

        Ort::Value &output_tensor = output_tensors[0];
        auto        output_info   = output_tensor.GetTensorTypeAndShapeInfo();
        auto        output_shape  = output_info.GetShape();
        if (output_shape.size() != 2)
        {
            continue;
        }

        const size_t num_classes = static_cast<size_t>(output_shape[1]);
        const float *preds       = output_tensor.GetTensorData<float>();
        std::vector<std::pair<std::string, float>> batch_res =
            postprocessor->decode(preds, batch_size, num_classes);

        for (size_t r = 0; r < batch_res.size(); ++r)
        {
            const size_t idx = indices[beg + r];
            cls_res[idx]     = batch_res[r];

            if (cls_res[idx].first.find("180") != std::string::npos &&
                cls_res[idx].second > cls_threshold)
            {
                cv::rotate(images[idx], images[idx], cv::ROTATE_180);
            }
        }
    }

    return {images, cls_res};
}

std::unique_ptr<Ort::Session> DirectionalClassifier::create_session() const
{
    return std::make_unique<Ort::Session>(env, model_filepath.c_str(),
                                          session_options);
}

std::vector<const char *> DirectionalClassifier::get_output_names(
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

std::vector<Ort::Value> DirectionalClassifier::run_inference(
    Ort::Session &session, const std::vector<const char *> &input_names,
    Ort::Value                      &input_tensor,
    const std::vector<const char *> &output_names) const
{
    return session.Run(Ort::RunOptions{nullptr}, input_names.data(),
                       &input_tensor, 1, output_names.data(),
                       output_names.size());
}
