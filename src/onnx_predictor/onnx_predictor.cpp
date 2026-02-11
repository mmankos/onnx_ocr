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
    if (det_filepath.empty() || !det_session)
    {
        std::cerr << "[ERROR][OnnxPredictor::predict] Detection model not "
                     "initialized.\n";
        return;
    }

    if (onnx_model_info.model.find(det_filepath) == onnx_model_info.model.end())
    {
        std::cerr << "[ERROR][OnnxPredictor::predict] Detection model info "
                     "missing.\n";
        return;
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

    Detector detector(env, session_options, memory_info, det_filepath,
                      onnx_model_info.model, keep_ratio, side_length_limit,
                      limit_type, det_threshold, det_box_threshold,
                      det_max_candidates, det_unclip_ratio, det_use_dilation);

    auto [classifier, cls_threshold_value] = prepare_cls();

    const std::vector<std::string> limit_types = {"min", "max", "resize_long"};
    const int                      limit_types_size = limit_types.size();
    int                            limit_type_num   = 1;

    if (limit_type == "best")
    {
        detector.set_limit_type(limit_types[0]);
        limit_type_num = limit_types_size;
    }

    for (int i = 0; i < limit_type_num; i++)
    {
        for (const std::pair<const std::string, std::shared_ptr<cv::Mat>>
                 &image : *images)
        {
            const cv::Mat original_image = image.second->clone();

            std::vector<Box> boxes = detector.run(image);

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

            std::string window_name =
                "Detection [" + detector.get_limit_type() + "]";
            if (cls_results.empty())
            {
                show_boxes(original_image, boxes, window_name);
            }
            else
            {
                show_boxes(original_image, boxes, window_name,
                           sorted_box_indices, cls_results,
                           cls_threshold_value);
            }

            // reset image to prevent duplicate pipeline application
            original_image.copyTo(*image.second);
        }

        if (i + 1 < limit_types_size)
        {
            detector.set_limit_type(limit_types[i + 1]);
        }
    }
}

std::pair<std::unique_ptr<DirectionalClassifier>, float>
OnnxPredictor::prepare_cls()
{
    if (cls_filepath.empty() || !cls_session.has_value() ||
        !cls_session.value())
    {
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

void OnnxPredictor::show_boxes(
    const cv::Mat &image, const std::vector<Box> &boxes,
    const std::string                                &window_name,
    const std::vector<size_t>                        &sorted_box_indices,
    const std::vector<std::pair<std::string, float>> &cls_results,
    float                                             cls_threshold) const
{
    if (boxes.empty())
    {
        return;
    }

    const cv::Scalar color_normal(0, 200, 0);
    const cv::Scalar color_rotated(0, 0, 255);
    const bool       has_cls = !cls_results.empty();

    cv::Mat display_image = image.clone();

    std::unordered_map<size_t, bool> rotated_map;
    if (has_cls)
    {
        const size_t count =
            std::min(sorted_box_indices.size(), cls_results.size());
        for (size_t i = 0; i < count; ++i)
        {
            const auto &res = cls_results[i];
            rotated_map[sorted_box_indices[i]] =
                res.first.find("180") != std::string::npos &&
                res.second > cls_threshold;
        }
    }

    for (size_t b = 0; b < boxes.size(); ++b)
    {
        std::vector<cv::Point> pts;
        pts.reserve(4);
        for (const auto &p : boxes[b])
        {
            pts.emplace_back(static_cast<int>(std::round(p.x)),
                             static_cast<int>(std::round(p.y)));
        }

        cv::Scalar color = color_normal;
        if (has_cls)
        {
            auto it = rotated_map.find(b);
            if (it != rotated_map.end() && it->second)
            {
                color = color_rotated;
            }
        }

        cv::polylines(display_image, pts, true, color, 2);
    }

    // legend strip below the image
    if (has_cls)
    {
        const int strip_h = 30;
        const int square  = 14;
        const int pad     = 10;

        cv::Mat strip(strip_h, display_image.cols, display_image.type(),
                      cv::Scalar(40, 40, 40));

        int x = pad;
        int y = (strip_h - square) / 2;

        cv::rectangle(strip, cv::Point(x, y), cv::Point(x + square, y + square),
                      color_normal, cv::FILLED);
        x += square + 5;
        cv::putText(strip, "normal", cv::Point(x, y + square - 2),
                    cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(220, 220, 220),
                    1);
        x += 60;

        cv::rectangle(strip, cv::Point(x, y), cv::Point(x + square, y + square),
                      color_rotated, cv::FILLED);
        x += square + 5;
        cv::putText(strip, "rotated", cv::Point(x, y + square - 2),
                    cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(220, 220, 220),
                    1);

        cv::vconcat(display_image, strip, display_image);
    }

    cv::imshow(window_name, display_image);
    cv::moveWindow(window_name, 200, 200);
    cv::waitKey(0);
    cv::destroyAllWindows();
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
