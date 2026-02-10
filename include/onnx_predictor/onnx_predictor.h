#pragma once

#include "yaml-cpp/yaml.h"
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <onnxruntime_cxx_api.h>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "loaders/config_loader.h"
#include "loaders/image_loader.h"
#include "ocr_pipeline/detection/detection.h"
#include "onnx_predictor/onnx_model_info.h"
#include "utils/utils.h"

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

    Ort::Env            env;
    Ort::SessionOptions session_options;
    Ort::MemoryInfo     memory_info{nullptr};
    Ort::Value          input_tensor{nullptr};

    std::string                                  det_filepath;
    std::unique_ptr<Ort::Session>                det_session{nullptr};
    std::string                                  rec_filepath;
    std::unique_ptr<Ort::Session>                rec_session{nullptr};
    std::string                                  cls_filepath;
    std::optional<std::unique_ptr<Ort::Session>> cls_session;
    OnnxModelInfo                                onnx_model_info;

    bool        keep_ratio;
    int         side_length_limit;
    std::string limit_type;
    std::string image_path;

    std::unique_ptr<Ort::Session>
    create_onnx_session(const std::string &filepath) const;

    void show_detection_result(const cv::Mat          &image,
                               const std::vector<Box> &boxes,
                               const Detector         &detector) const;
};
