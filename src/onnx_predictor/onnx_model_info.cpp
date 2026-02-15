#include "onnx_predictor/onnx_model_info.h"

#include <iostream>
#include <onnxruntime_cxx_api.h>

template <typename T> void print_vector(const std::vector<T> &v)
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

void OnnxModelInfo::fill(const Ort::Session &session,
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
        std::vector<int64_t> image_shape =
            (shape.size() == 4 && shape[2] > 0 && shape[3] > 0)
                ? std::vector<int64_t>{shape[2], shape[3]}
                : std::vector<int64_t>();

        model[model_name] = OnnxModelInputInfo{
            std::move(name.get()), std::move(shape),
            std::make_shared<std::vector<int64_t>>(image_shape)};
    }
}

void OnnxModelInfo::print() const
{
    std::cout << "OnnxModelInfo:\n";

    for (const auto &entry : model)
    {
        std::cout << "  Model: " << entry.first << "\n";
        std::cout << "    Input name: " << entry.second.name << "\n";
        std::cout << "    Shape: ";
        print_vector(entry.second.shape);
        std::cout << "\n";
    }
}
