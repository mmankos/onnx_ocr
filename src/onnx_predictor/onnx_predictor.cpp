#include "onnx_predictor/onnx_predictor.h"

OnnxPredictor::OnnxPredictor(const std::string &config_filepath)
{
    try
    {
        env = Ort::Env{ORT_LOGGING_LEVEL_WARNING, "Default"};
    }
    catch (const Ort::Exception &e)
    {
        std::cerr << "[ERROR][OnnxPredictor::OnnxPredictor] Failed to create "
                     "ONNX Env: "
                  << e.what() << "\n";
        throw std::runtime_error("Failed to create ONNX Env");
    }

    session_options.SetInterOpNumThreads(1);
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL);

    memory_info =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    config_loader = std::make_unique<ConfigLoader>(config_filepath);

    det_filepath =
        config_loader->get<std::string>(DET_ONNX_MODEL_FILEPATH).value_or("");
    rec_filepath =
        config_loader->get<std::string>(REC_ONNX_MODEL_FILEPATH).value_or("");
    cls_filepath =
        config_loader->get<std::string>(CLS_ONNX_MODEL_FILEPATH).value_or("");

    if (det_filepath.empty() ||
        !(det_session = create_onnx_session(det_filepath)))
    {
        throw std::runtime_error("Failed to load ONNX model");
    }
    onnx_model_info.fill(*det_session, det_filepath);

    if (rec_filepath.empty() ||
        !(rec_session = create_onnx_session(rec_filepath)))
    {
        throw std::runtime_error("Failed to load ONNX model");
    }
    onnx_model_info.fill(*rec_session, rec_filepath);

    if (!cls_filepath.empty())
    {
        std::unique_ptr<Ort::Session> session =
            create_onnx_session(cls_filepath);
        if (session)
        {
            onnx_model_info.fill(*session, cls_filepath);
            cls_session = std::move(session);
        }
    }

    side_length_limit =
        config_loader->get<int>(SIDE_LENGTH_LIMIT).value_or(760);
    limit_type = config_loader->get<std::string>(LIMIT_TYPE).value_or("best");
    image_path = config_loader->get<std::string>(IMAGE_PATH).value_or(".");

    image_loader = std::make_unique<ImageLoader>(image_path);
    images       = image_loader->get_images();
    std::cout << "\n[INFO] " << images->size() << " images loaded."
              << std::endl;

    keep_ratio = false;
}

void OnnxPredictor::predict()
{
    auto detector = prepare_det();
    if (!detector)
    {
        std::cerr << "[ERROR][OnnxPredictor::predict] Detection model not "
                     "ready.\n";
        return;
    }

    auto [classifier, cls_threshold_value] = prepare_cls();
    auto [recognizer, drop_score]          = prepare_rec();

    const std::vector<std::string> limit_types = {"min", "max", "resize_long"};
    const int                      limit_types_size = limit_types.size();
    int                            limit_type_num   = 1;

    if (limit_type == "best")
    {
        detector->set_limit_type(limit_types[0]);
        limit_type_num = limit_types_size;
    }

    for (int i = 0; i < limit_type_num; i++)
    {
        for (const std::pair<const std::string, std::shared_ptr<cv::Mat>>
                 &image : *images)
        {
            const cv::Mat original_image = image.second->clone();

            std::vector<Box> boxes = detector->run(image);

            std::vector<size_t>  sorted_box_indices = sort_box_indices(boxes);
            std::vector<cv::Mat> text_crops;

            for (size_t idx = 0; idx < sorted_box_indices.size(); ++idx)
            {
                size_t  i    = sorted_box_indices[idx];
                cv::Mat crop = image_box_crop(original_image, boxes[i]);
                text_crops.push_back(crop);
            }

            std::vector<std::pair<std::string, float>> cls_results;
            if (classifier)
            {
                auto cls_output = classifier->run(text_crops);
                text_crops      = std::move(cls_output.first);
                cls_results     = std::move(cls_output.second);
            }

            // build rec text map: box_index -> recognized text
            std::unordered_map<size_t, std::string> rec_text_map;
            if (recognizer)
            {
                std::vector<std::pair<std::string, float>> rec_results =
                    recognizer->run(text_crops);

                std::vector<Box>                           filter_boxes;
                std::vector<std::pair<std::string, float>> filter_rec_res;
                filter_boxes.reserve(boxes.size());
                filter_rec_res.reserve(rec_results.size());

                const size_t rec_count =
                    std::min(rec_results.size(), sorted_box_indices.size());
                for (size_t r = 0; r < rec_count; ++r)
                {
                    const size_t box_idx = sorted_box_indices[r];
                    if (box_idx >= boxes.size())
                    {
                        continue;
                    }

                    if (rec_results[r].second >= drop_score)
                    {
                        filter_boxes.push_back(boxes[box_idx]);
                        filter_rec_res.push_back(rec_results[r]);
                        rec_text_map[box_idx] = rec_results[r].first;
                    }
                }

                std::cout << "\n[REC][" << detector->get_limit_type() << "] "
                          << image.first << "\n";
                if (filter_rec_res.empty())
                {
                    std::cout << "  (no results above drop_score)\n";
                }
                else
                {
                    for (size_t r = 0; r < filter_rec_res.size(); ++r)
                    {
                        std::cout << "  " << r << ": "
                                  << filter_rec_res[r].first << " ("
                                  << filter_rec_res[r].second << ")\n";
                    }
                }
            }

            std::string window_name =
                "Detection + Directional Classification + Recognition [" +
                detector->get_limit_type() + "]";
            show_boxes(original_image, boxes, window_name, sorted_box_indices,
                       cls_results, cls_threshold_value, rec_text_map);

            // reset image to prevent duplicate pipeline application
            original_image.copyTo(*image.second);
        }

        if (i + 1 < limit_types_size)
        {
            detector->set_limit_type(limit_types[i + 1]);
        }
    }
}

std::unique_ptr<Detector> OnnxPredictor::prepare_det()
{
    if (det_filepath.empty() || !det_session)
    {
        std::cerr << "[ERROR][OnnxPredictor::prepare_det] Detection model not "
                     "initialized.\n";
        return nullptr;
    }

    if (onnx_model_info.model.find(det_filepath) == onnx_model_info.model.end())
    {
        std::cerr << "[ERROR][OnnxPredictor::prepare_det] Detection model info "
                     "missing.\n";
        return nullptr;
    }

    const float det_threshold =
        config_loader->get<float>(DET_THRESH).value_or(0.3f);
    const float det_box_threshold =
        config_loader->get<float>(DET_BOX_THRESH).value_or(0.6f);
    const int det_max_candidates =
        config_loader->get<int>(DET_MAX_CANDIDATES).value_or(1000);
    const float det_unclip_ratio =
        config_loader->get<float>(DET_UNCLIP_RATIO).value_or(1.5f);
    const bool det_use_dilation =
        config_loader->get<bool>(DET_USE_DILATION).value_or(false);

    return std::make_unique<Detector>(
        env, session_options, memory_info, det_filepath, onnx_model_info.model,
        keep_ratio, side_length_limit, limit_type, det_threshold,
        det_box_threshold, det_max_candidates, det_unclip_ratio,
        det_use_dilation);
}

std::pair<std::unique_ptr<DirectionalClassifier>, float>
OnnxPredictor::prepare_cls()
{
    if (cls_filepath.empty() || !cls_session.has_value() ||
        !cls_session.value())
    {
        return {nullptr, 0.0f};
    }

    if (onnx_model_info.model.find(cls_filepath) == onnx_model_info.model.end())
    {
        std::cerr << "[WARN][OnnxPredictor::prepare_cls] Classification model "
                     "info missing; skipping cls.\n";
        return {nullptr, 0.0f};
    }

    std::vector<int> config_image_shape =
        config_loader->get<std::vector<int>>(CLS_IMAGE_SHAPE)
            .value_or(std::vector<int>{});

    auto             model_it = onnx_model_info.model.find(cls_filepath);
    std::vector<int> model_image_shape;
    if (model_it != onnx_model_info.model.end())
    {
        const auto &shape = model_it->second.shape;
        if (shape.size() == 4 && shape[1] > 0 && shape[2] > 0 && shape[3] > 0)
        {
            model_image_shape = {static_cast<int>(shape[1]),
                                 static_cast<int>(shape[2]),
                                 static_cast<int>(shape[3])};
        }
    }

    std::vector<int> cls_image_shape;
    if (model_image_shape.size() == 3)
    {
        cls_image_shape = model_image_shape;
        if (config_image_shape.size() == 3 &&
            config_image_shape != model_image_shape)
        {
            std::cerr << "[WARN][OnnxPredictor::prepare_cls] cls_image_shape "
                         "does not match model input shape; using model "
                         "shape instead.\n";
        }
    }
    else if (config_image_shape.size() == 3)
    {
        cls_image_shape = config_image_shape;
    }
    else
    {
        cls_image_shape = {3, 80, 160};
        std::cerr << "[WARN][OnnxPredictor::prepare_cls] cls_image_shape not "
                     "set and model input shape is dynamic; using "
                     "default 3x80x160.\n";
    }

    const int cls_batch_size =
        config_loader->get<int>(CLS_BATCH_SIZE).value_or(6);
    const float cls_threshold =
        config_loader->get<float>(CLS_THRESH).value_or(0.9f);
    std::vector<std::string> label_list =
        config_loader->get<std::vector<std::string>>(CLS_LABEL_LIST)
            .value_or(std::vector<std::string>{"0", "180"});

    auto classifier = std::make_unique<DirectionalClassifier>(
        env, session_options, memory_info, cls_filepath, onnx_model_info.model,
        cls_image_shape, cls_batch_size, cls_threshold, std::move(label_list));

    return {std::move(classifier), cls_threshold};
}

std::pair<std::unique_ptr<TextRecognizer>, float> OnnxPredictor::prepare_rec()
{
    float drop_score = config_loader->get<float>(DROP_SCORE).value_or(0.82f);

    if (rec_filepath.empty() || !rec_session)
    {
        return {nullptr, drop_score};
    }

    if (onnx_model_info.model.find(rec_filepath) == onnx_model_info.model.end())
    {
        std::cerr << "[ERROR][OnnxPredictor::prepare_rec] Recognition model "
                     "info missing.\n";
        return {nullptr, drop_score};
    }

    std::vector<int> rec_image_shape =
        config_loader->get<std::vector<int>>(REC_IMAGE_SHAPE)
            .value_or(std::vector<int>{});

    auto model_it = onnx_model_info.model.find(rec_filepath);
    if (model_it != onnx_model_info.model.end())
    {
        const auto &shape = model_it->second.shape;
        if (shape.size() == 4 && shape[1] > 0 && shape[2] > 0)
        {
            rec_image_shape = {static_cast<int>(shape[1]),
                               static_cast<int>(shape[2]),
                               static_cast<int>(shape[3] > 0 ? shape[3]
                                                : rec_image_shape.size() == 3
                                                    ? rec_image_shape[2]
                                                    : 320)};
        }
    }

    if (rec_image_shape.size() != 3)
    {
        rec_image_shape = {3, 48, 320};
    }

    const int rec_batch_size =
        config_loader->get<int>(REC_BATCH_SIZE).value_or(6);
    const std::string rec_char_dict_path =
        config_loader->get<std::string>(REC_CHAR_DICT_PATH)
            .value_or("data/dict.txt");
    const bool use_space_char =
        config_loader->get<bool>(REC_USE_SPACE_CHAR).value_or(true);
    const float rec_norm_scale =
        config_loader->get<float>(REC_NORM_SCALE).value_or(1.0f / 255.0f);
    std::vector<float> rec_norm_mean =
        config_loader->get<std::vector<float>>(REC_NORM_MEAN)
            .value_or(std::vector<float>{0.5f, 0.5f, 0.5f});
    std::vector<float> rec_norm_std =
        config_loader->get<std::vector<float>>(REC_NORM_STD)
            .value_or(std::vector<float>{0.5f, 0.5f, 0.5f});

    auto recognizer = std::make_unique<TextRecognizer>(
        env, session_options, memory_info, rec_filepath, onnx_model_info.model,
        rec_image_shape, rec_batch_size, rec_char_dict_path, use_space_char,
        rec_norm_scale, std::move(rec_norm_mean), std::move(rec_norm_std));

    return {std::move(recognizer), drop_score};
}

std::unique_ptr<Ort::Session>
OnnxPredictor::create_onnx_session(const std::string &filepath) const
{
    try
    {
        std::unique_ptr<Ort::Session> session = std::make_unique<Ort::Session>(
            env, filepath.c_str(), session_options);

        std::cout << "\n[SUCCESS] " << filepath << " loaded successfully!\n";

        return session;
    }
    catch (const Ort::Exception &e)
    {
        std::cerr << "[ERROR][OnnxPredictor::create_onnx_session] Failed to "
                     "load ONNX model:\n"
                  << "    " << filepath << " → "
                  << (std::filesystem::exists(filepath) ? "found" : "MISSING")
                  << "\n"
                  << "    Error: " << e.what() << " (code "
                  << e.GetOrtErrorCode() << ")\n";

        return nullptr;
    }
}
