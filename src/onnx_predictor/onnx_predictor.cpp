#include "onnx_predictor/onnx_predictor.h"

OnnxPredictor::OnnxPredictor(const std::string &config_filepath)
{
    try
    {
        env = Ort::Env{ORT_LOGGING_LEVEL_WARNING, "Default"};
    }
    catch (const Ort::Exception &e)
    {
        std::cerr << "[ERROR] Failed to create ONNX Env: " << e.what() << "\n";
        throw std::runtime_error("Failed to create ONNX Env");
    }

    sessionOptions.SetInterOpNumThreads(1);
    sessionOptions.SetIntraOpNumThreads(1);
    sessionOptions.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL);

    config_loader = std::make_unique<ConfigLoader>(config_filepath);

    std::optional<std::string> det_filepath =
        config_loader->get<std::string>(DET_ONNX_MODEL_FILEPATH);
    std::optional<std::string> rec_filepath =
        config_loader->get<std::string>(REC_ONNX_MODEL_FILEPATH);
    std::optional<std::string> cls_filepath =
        config_loader->get<std::string>(CLS_ONNX_MODEL_FILEPATH);

    if (!det_filepath || !(det_session = createOnnxSession(*det_filepath)))
    {
        throw std::runtime_error("Failed to load ONNX model");
    }
    fillModelInfo(*det_session);

    if (!rec_filepath || !(rec_session = createOnnxSession(*rec_filepath)))
    {
        throw std::runtime_error("Failed to load ONNX model");
    }
    fillModelInfo(*rec_session);

    if (cls_filepath && (cls_session = createOnnxSession(*cls_filepath)))
    {
        fillModelInfo(**cls_session);
    }

    side_length_limit =
        config_loader->get<int>(SIDE_LENGTH_LIMIT).value_or(760);
    limit_type = config_loader->get<std::string>(LIMIT_TYPE).value_or("best");

    image_path = config_loader->get<std::string>(IMAGE_PATH).value_or(".");

    image_loader = std::make_unique<ImageLoader>(image_path);
    images       = image_loader->get_images();
    std::cout << "\n[INFO] " << images->size() << " images loaded."
              << std::endl;
}

void OnnxPredictor::predict() {}

std::unique_ptr<Ort::Session>
OnnxPredictor::createOnnxSession(const std::string &filepath) const
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
        std::cerr << "[ERROR] Failed to load ONNX model:\n"
                  << "    " << filepath << " → "
                  << (std::filesystem::exists(filepath) ? "found" : "MISSING")
                  << "\n"
                  << "    Error: " << e.what() << " (code "
                  << e.GetOrtErrorCode() << ")\n";

        return nullptr;
    }
}

void OnnxPredictor::fillModelInfo(const Ort::Session &session)
{
    Ort::AllocatorWithDefaultOptions allocator;
    size_t                           num_inputs = session.GetInputCount();

    for (size_t i = 0; i < num_inputs; ++i)
    {
        auto name        = session.GetInputNameAllocated(i, allocator);
        auto type_info   = session.GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        auto shape       = tensor_info.GetShape();

        std::cout << "Input " << i << ": name='" << name.get() << "' shape=[";
        for (size_t j = 0; j < shape.size(); ++j)
        {
            std::cout << (shape[j] == -1 ? "dynamic"
                                         : std::to_string(shape[j]));
            if (j < shape.size() - 1)
                std::cout << ",";
        }
        std::cout << "]\n";

        onnx_model_info.inputs.push_back(
            {std::move(name.get()), std::move(shape)});
    }
}

template <typename T>
void OnnxPredictor::print_vector(const std::vector<T> &v) const
{
    std::cout << "[";
    for (size_t i = 0; i < v.size(); ++i)
    {
        std::cout << v[i];
        if (i + 1 < v.size())
            std::cout << ", ";
    }
    std::cout << "]";
}

void OnnxPredictor::print_onnx_model_info() const
{
    std::cout << "OnnxModelInfo:\n";
    for (const auto &input : onnx_model_info.inputs)
    {
        std::cout << "  Input name: " << input.name << "\n";
        std::cout << "  Shape: ";
        print_vector(input.shape);
        std::cout << "\n";
    }
}
