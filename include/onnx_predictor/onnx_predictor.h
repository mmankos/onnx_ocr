#pragma once
#include "yaml-cpp/yaml.h"
#include <iostream>
#include <memory>
#include <onnxruntime_cxx_api.h>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "loaders/config_loader.h"
#include "loaders/image_loader.h"

struct OnnxModelInputInfo
{
    std::string          name;
    std::vector<int64_t> shape;
};

struct OnnxModelInfo
{
    std::vector<OnnxModelInputInfo> inputs;
};

struct OnnxModelOutputInfo
{};

class OnnxPredictor
{
  public:
    OnnxPredictor(const std::string &config_filepath);

    void predict();

  private:
    std::unique_ptr<ConfigLoader>                             config_loader;
    std::unique_ptr<ImageLoader>                              image_loader;
    std::shared_ptr<std::unordered_map<std::string, cv::Mat>> images;
    OnnxModelInfo                                             onnx_model_info;
    Ort::Env                                                  env;
    Ort::SessionOptions                                       sessionOptions;
    std::unique_ptr<Ort::Session>                det_session{nullptr};
    std::unique_ptr<Ort::Session>                rec_session{nullptr};
    std::optional<std::unique_ptr<Ort::Session>> cls_session{nullptr};
    int                                          side_length_limit;
    std::string                                  limit_type;
    std::string                                  image_path;

    std::unique_ptr<Ort::Session>
         createOnnxSession(const std::string &filepath) const;
    void fillModelInfo(const Ort::Session &session);
    template <typename T> void print_vector(const std::vector<T> &v) const;
    void                       print_onnx_model_info() const;
};
