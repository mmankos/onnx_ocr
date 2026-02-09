#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

#include "detection/postprocess/detection_postprocessor.h"
#include "detection/preprocess/detection_preprocessor.h"
#include "onnx_predictor/onnx_predictor.h"

struct OnnxModelInputInfo;

class Detector
{
  public:
    Detector(Ort::Env &env, const Ort::SessionOptions &session_options,
             Ort::MemoryInfo &memory_info, const std::string &det_filepath,
             const std::unordered_map<std::string, OnnxModelInputInfo> &models,
             bool keep_ratio, int side_length_limit,
             const std::string &limit_type);

    const std::string &get_limit_type() const;
    void               set_limit_type(std::string limit_type);
    std::vector<std::array<cv::Point2f, 4>>
    run(const std::pair<const std::string, std::shared_ptr<cv::Mat>> &image);

  private:
    Ort::Env                  &env;
    const Ort::SessionOptions &session_options;
    Ort::MemoryInfo           &memory_info;

    std::string                                                det_filepath;
    const std::unordered_map<std::string, OnnxModelInputInfo> &models;

    bool        keep_ratio;
    int         side_length_limit;
    std::string limit_type;

    std::unique_ptr<DetectionPostprocessor> postprocessor;
};
