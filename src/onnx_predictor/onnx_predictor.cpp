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

    sessionOptions.SetInterOpNumThreads(1);
    sessionOptions.SetIntraOpNumThreads(1);
    sessionOptions.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL);

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
    fill_model_info(*det_session, det_filepath);

    if (rec_filepath.empty() ||
        !(rec_session = create_onnx_session(rec_filepath)))
    {
        throw std::runtime_error("Failed to load ONNX model");
    }
    fill_model_info(*rec_session, rec_filepath);

    if (!cls_filepath.empty() &&
        (cls_session = create_onnx_session(cls_filepath)))
    {
        fill_model_info(**cls_session, cls_filepath);
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
    for (const std::pair<const std::string, std::shared_ptr<cv::Mat>> &image :
         *images)
    {
        std::unique_ptr<DetectionPreprocessor> detection_preprocessor =
            std::make_unique<DetectionPreprocessor>(
                keep_ratio, side_length_limit, limit_type,
                onnx_model_info.model[det_filepath].image_shape, image);

        detection_preprocessor->preprocess();
    }
}

std::unique_ptr<Ort::Session>
OnnxPredictor::create_onnx_session(const std::string &filepath) const
{
    try
    {
        std::unique_ptr<Ort::Session> session = std::make_unique<Ort::Session>(
            env, filepath.c_str(), sessionOptions);

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

void OnnxPredictor::fill_model_info(const Ort::Session &session,
                                    const std::string  &model_name)
{
    Ort::AllocatorWithDefaultOptions allocator;
    size_t                           num_inputs = session.GetInputCount();

    for (size_t i = 0; i < num_inputs; ++i)
    {
        auto name        = session.GetInputNameAllocated(i, allocator);
        auto type_info   = session.GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

        std::vector<int64_t> shape = tensor_info.GetShape();
        std::optional<std::pair<int64_t, int64_t>> image_shape =
            (shape.size() == 4 && shape[2] > 0 && shape[3] > 0)
                ? std::optional<std::pair<int64_t, int64_t>>{std::make_pair(
                      shape[2], shape[3])}
                : std::nullopt;

        onnx_model_info.model[model_name] = OnnxModelInputInfo{
            std::move(name.get()), std::move(shape), std::move(image_shape)};
    }
}

template <typename T>
void OnnxPredictor::print_vector(const std::vector<T> &vector) const
{
    std::cout << "[";
    for (size_t i = 0; i < vector.size(); ++i)
    {
        std::cout << vector[i];
        if (i + 1 < vector.size())
            std::cout << ", ";
    }
    std::cout << "]";
}

void OnnxPredictor::print_onnx_model_info() const
{
    std::cout << "OnnxModelInfo:\n";

    for (const auto &model : onnx_model_info.model)
    {
        std::cout << "  Model: " << model.first << "\n";
        std::cout << "    Input name: " << model.second.name << "\n";
        std::cout << "    Shape: ";
        print_vector(model.second.shape);
        std::cout << "\n";
    }
}
