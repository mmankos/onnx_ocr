#include "ocr_pipeline/recognition/recognition.h"

TextRecognizer::TextRecognizer(
    Ort::Env &env, const Ort::SessionOptions &session_options,
    Ort::MemoryInfo &memory_info, const std::string &model_filepath,
    std::unordered_map<std::string, OnnxModelInputInfo> &models,
    std::vector<int> rec_image_shape, int rec_batch_size,
    const std::string &rec_char_dict_path, bool use_space_char,
    float rec_norm_scale, std::vector<float> rec_norm_mean,
    std::vector<float> rec_norm_std)
    : env(env), session_options(session_options), memory_info(memory_info),
      model_filepath(model_filepath), models(models),
      rec_image_shape(std::move(rec_image_shape)),
      rec_batch_size(rec_batch_size), model_width(0)
{
    auto model_it = models.find(model_filepath);
    if (model_it != models.end())
    {
        const auto &shape = model_it->second.shape;
        if (shape.size() == 4 && shape[3] > 0)
        {
            model_width = static_cast<int>(shape[3]);
        }
    }

    preprocessor = std::make_unique<RecognitionPreprocess>(
        this->rec_image_shape, model_width, rec_norm_scale,
        std::move(rec_norm_mean), std::move(rec_norm_std));
    postprocessor = std::make_unique<RecognitionPostprocess>(rec_char_dict_path,
                                                             use_space_char);
}

std::vector<std::pair<std::string, float>>
TextRecognizer::run(const std::vector<cv::Mat> &image_list)
{
    const size_t image_num = image_list.size();
    if (image_num == 0)
    {
        return {};
    }

    std::vector<size_t> indices =
        preprocessor->sort_indices_by_aspect(image_list);
    std::vector<std::pair<std::string, float>> rec_res(image_num, {"", 0.0f});

    std::unique_ptr<Ort::Session>        session = create_session();
    std::vector<Ort::AllocatedStringPtr> output_name_allocs;
    std::vector<const char *>            output_names =
        get_output_names(*session, output_name_allocs);

    for (size_t beg = 0; beg < image_num; beg += rec_batch_size)
    {
        size_t end =
            std::min(image_num, beg + static_cast<size_t>(rec_batch_size));
        std::vector<float> wh_ratios;
        wh_ratios.reserve(end - beg);

        float max_wh_ratio =
            static_cast<float>(rec_image_shape[2]) / rec_image_shape[1];
        for (size_t i = beg; i < end; ++i)
        {
            const cv::Mat &img      = image_list[indices[i]];
            float          wh_ratio = img.cols / static_cast<float>(img.rows);
            max_wh_ratio            = std::max(max_wh_ratio, wh_ratio);
            wh_ratios.push_back(wh_ratio);
        }

        const int target_width =
            preprocessor->compute_target_width(max_wh_ratio);
        const int c = rec_image_shape[0];
        const int h = rec_image_shape[1];

        std::vector<cv::Mat> batch;
        batch.reserve(end - beg);
        for (size_t i = beg; i < end; ++i)
        {
            cv::Mat norm_img = preprocessor->resize_and_normalize(
                image_list[indices[i]], target_width);
            batch.push_back(norm_img);
        }

        const int          batch_size = static_cast<int>(batch.size());
        std::vector<float> batch_data(batch_size * c * h * target_width);
        for (int i = 0; i < batch_size; ++i)
        {
            std::memcpy(batch_data.data() + i * c * h * target_width,
                        batch[i].ptr<float>(0),
                        sizeof(float) * c * h * target_width);
        }

        std::vector<int64_t> input_shape  = {batch_size, c, h, target_width};
        Ort::Value           input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, batch_data.data(), batch_data.size(),
            input_shape.data(), input_shape.size());

        std::vector<const char *> input_names;
        auto                      model_it = models.find(model_filepath);
        if (model_it == models.end())
        {
            return rec_res;
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
        if (output_shape.size() != 3)
        {
            continue;
        }

        const size_t expected_classes = postprocessor->num_classes();
        const float *predictions      = output_tensor.GetTensorData<float>();

        size_t             time_steps  = 0;
        size_t             num_classes = 0;
        std::vector<float> reordered;
        const bool         has_batch0 = static_cast<size_t>(output_shape[0]) ==
                                static_cast<size_t>(batch_size);
        const bool has_batch1 = static_cast<size_t>(output_shape[1]) ==
                                static_cast<size_t>(batch_size);

        if (has_batch0 &&
            static_cast<size_t>(output_shape[2]) == expected_classes)
        {
            time_steps  = static_cast<size_t>(output_shape[1]);
            num_classes = expected_classes;
        }
        else if (has_batch0 &&
                 static_cast<size_t>(output_shape[1]) == expected_classes)
        {
            time_steps  = static_cast<size_t>(output_shape[2]);
            num_classes = expected_classes;
            reordered.resize(batch_size * time_steps * num_classes);
            for (size_t b = 0; b < static_cast<size_t>(batch_size); ++b)
            {
                for (size_t c = 0; c < num_classes; ++c)
                {
                    for (size_t t = 0; t < time_steps; ++t)
                    {
                        const size_t src =
                            (b * num_classes + c) * time_steps + t;
                        const size_t dst =
                            (b * time_steps + t) * num_classes + c;
                        reordered[dst] = predictions[src];
                    }
                }
            }
            predictions = reordered.data();
        }
        else if (has_batch1 &&
                 static_cast<size_t>(output_shape[2]) == expected_classes)
        {
            time_steps  = static_cast<size_t>(output_shape[0]);
            num_classes = expected_classes;
            reordered.resize(batch_size * time_steps * num_classes);
            for (size_t t = 0; t < time_steps; ++t)
            {
                for (size_t b = 0; b < static_cast<size_t>(batch_size); ++b)
                {
                    for (size_t c = 0; c < num_classes; ++c)
                    {
                        const size_t src =
                            (t * static_cast<size_t>(batch_size) + b) *
                                num_classes +
                            c;
                        const size_t dst =
                            (b * time_steps + t) * num_classes + c;
                        reordered[dst] = predictions[src];
                    }
                }
            }
            predictions = reordered.data();
        }
        else
        {
            time_steps  = static_cast<size_t>(output_shape[1]);
            num_classes = static_cast<size_t>(output_shape[2]);
        }

        std::vector<std::pair<std::string, float>> batch_res =
            postprocessor->decode(predictions, batch_size, time_steps,
                                  num_classes);

        for (size_t r = 0; r < batch_res.size(); ++r)
        {
            rec_res[indices[beg + r]] = batch_res[r];
        }
    }

    return rec_res;
}

std::unique_ptr<Ort::Session> TextRecognizer::create_session() const
{
    return std::make_unique<Ort::Session>(env, model_filepath.c_str(),
                                          session_options);
}

std::vector<const char *> TextRecognizer::get_output_names(
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

std::vector<Ort::Value> TextRecognizer::run_inference(
    Ort::Session &session, const std::vector<const char *> &input_names,
    Ort::Value                      &input_tensor,
    const std::vector<const char *> &output_names) const
{
    return session.Run(Ort::RunOptions{nullptr}, input_names.data(),
                       &input_tensor, 1, output_names.data(),
                       output_names.size());
}
