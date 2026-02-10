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

    const float det_thresh =
        config_loader->get<float>(DET_THRESH).value_or(0.3f);
    const float det_box_thresh =
        config_loader->get<float>(DET_BOX_THRESH).value_or(0.6f);
    const int det_max_candidates =
        config_loader->get<int>(DET_MAX_CANDIDATES).value_or(1000);
    const float det_unclip_ratio =
        config_loader->get<float>(DET_UNCLIP_RATIO).value_or(1.5f);
    const bool det_use_dilation =
        config_loader->get<bool>(DET_USE_DILATION).value_or(false);

    Detector detector(env, session_options, memory_info, det_filepath,
                      onnx_model_info.model, keep_ratio, side_length_limit,
                      limit_type, det_thresh, det_box_thresh,
                      det_max_candidates, det_unclip_ratio, det_use_dilation);

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
            show_detection_result(original_image, boxes, detector);

            // reset image to prevent duplicate pipeline application
            original_image.copyTo(*image.second);
        }

        if (i + 1 < limit_types_size)
        {
            detector.set_limit_type(limit_types[i + 1]);
        }
    }
}

void OnnxPredictor::show_detection_result(const cv::Mat          &image,
                                          const std::vector<Box> &boxes,
                                          const Detector &detector) const
{
    cv::Mat display_image = image.clone();
    for (const auto &box : boxes)
    {
        std::vector<cv::Point> pts;
        pts.reserve(4);
        for (const auto &p : box)
        {
            pts.emplace_back(static_cast<int>(std::round(p.x)),
                             static_cast<int>(std::round(p.y)));
        }
        cv::polylines(display_image, pts, true, cv::Scalar(0, 255, 0), 2);
    }

    std::string window_name =
        "Detection limit type: " + detector.get_limit_type();
    cv::imshow(window_name, display_image);
    cv::moveWindow(window_name, 200, 200);
    cv::waitKey(0);
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
