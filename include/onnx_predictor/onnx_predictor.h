#pragma once
#include "yaml-cpp/yaml.h"
#include <cstdint>
#include <iostream>
#include <memory>
#include <onnxruntime_cxx_api.h>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "loaders/config_loader.h"
#include "loaders/image_loader.h"

#include "detection/preprocess/detection_preprocessor.h"

struct OnnxModelInputInfo
{
    std::string                                name;
    std::vector<int64_t>                       shape;
    std::optional<std::pair<int64_t, int64_t>> image_shape;
};

struct OnnxModelInfo
{
    std::unordered_map<std::string, OnnxModelInputInfo> model;
};

class OnnxPredictor
{
  public:
    OnnxPredictor(const std::string &config_filepath);

    void predict();

  private:
    std::unique_ptr<ConfigLoader> config_loader;
    std::unique_ptr<ImageLoader>  image_loader;
    std::shared_ptr<std::unordered_map<std::string, std::shared_ptr<cv::Mat>>>
        images;

    Ort::Env                                     env;
    Ort::SessionOptions                          sessionOptions;
    std::string                                  det_filepath;
    std::unique_ptr<Ort::Session>                det_session{nullptr};
    std::string                                  rec_filepath;
    std::unique_ptr<Ort::Session>                rec_session{nullptr};
    std::string                                  cls_filepath;
    std::optional<std::unique_ptr<Ort::Session>> cls_session{nullptr};
    OnnxModelInfo                                onnx_model_info;

    bool        keep_ratio;
    int         side_length_limit;
    std::string limit_type;
    std::string image_path;

    std::unique_ptr<Ort::Session>
         create_onnx_session(const std::string &filepath) const;
    void fill_model_info(const Ort::Session &session,
                         const std::string  &model_name);
    template <typename T> void print_vector(const std::vector<T> &v) const;
    void                       print_onnx_model_info() const;
};
