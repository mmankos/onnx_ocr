#pragma once

#include <memory>
#include <onnxruntime_cxx_api.h>
#include <string>
#include <unordered_map>
#include <vector>

struct OnnxModelInputInfo
{
    std::string                           name;
    std::vector<int64_t>                  shape;
    std::shared_ptr<std::vector<int64_t>> image_shape;
};

struct OnnxModelInfo
{
    std::unordered_map<std::string, OnnxModelInputInfo> model;

    void fill(const Ort::Session &session, const std::string &model_name);
    void print() const;
};
