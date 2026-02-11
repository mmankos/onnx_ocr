#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

#include "ocr_pipeline/directional_classification/postprocess/directional_classification_postprocess.h"
#include "ocr_pipeline/directional_classification/preprocess/directional_classification_preprocess.h"
#include "onnx_predictor/onnx_model_info.h"
#include "utils/utils.h"

class DirectionalClassifier
{
  public:
    DirectionalClassifier(
        Ort::Env &env, const Ort::SessionOptions &session_options,
        Ort::MemoryInfo &memory_info, const std::string &model_filepath,
        std::unordered_map<std::string, OnnxModelInputInfo> &models,
        std::vector<int> cls_image_shape, int cls_batch_size,
        float cls_threshold, std::vector<std::string> label_list);

    std::pair<std::vector<cv::Mat>, std::vector<std::pair<std::string, float>>>
    run(const std::vector<cv::Mat> &text_crops);

  private:
    std::unique_ptr<Ort::Session> create_session() const;
    std::vector<const char *>     get_output_names(
            Ort::Session                         &session,
            std::vector<Ort::AllocatedStringPtr> &output_name_allocs) const;
    std::vector<Ort::Value> run_inference(
        Ort::Session &session, const std::vector<const char *> &input_names,
        Ort::Value                      &input_tensor,
        const std::vector<const char *> &output_names) const;

    Ort::Env                  &env;
    const Ort::SessionOptions &session_options;
    Ort::MemoryInfo           &memory_info;

    std::string                                          model_filepath;
    std::unordered_map<std::string, OnnxModelInputInfo> &models;

    std::vector<int> cls_image_shape;
    int              cls_batch_size;
    float            cls_threshold;

    std::unique_ptr<DirectionalClassificationPreprocess>  preprocessor;
    std::unique_ptr<DirectionalClassificationPostprocess> postprocessor;
};
