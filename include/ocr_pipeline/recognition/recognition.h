#pragma once

#include <algorithm>
#include <cstring>
#include <memory>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "ocr_pipeline/recognition/postprocess/recognition_postprocess.h"
#include "ocr_pipeline/recognition/preprocess/recognition_preprocess.h"
#include "onnx_predictor/onnx_model_info.h"

class TextRecognizer
{
  public:
    TextRecognizer(Ort::Env &env, const Ort::SessionOptions &session_options,
                   Ort::MemoryInfo   &memory_info,
                   const std::string &model_filepath,
                   std::unordered_map<std::string, OnnxModelInputInfo> &models,
                   std::vector<int> rec_image_shape, int rec_batch_size,
                   const std::string &rec_char_dict_path, bool use_space_char,
                   float rec_norm_scale, std::vector<float> rec_norm_mean,
                   std::vector<float> rec_norm_std);

    std::vector<std::pair<std::string, float>>
    run(const std::vector<cv::Mat> &img_list);

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

    std::vector<int> rec_image_shape;
    int              rec_batch_size;
    int              model_width;

    std::unique_ptr<RecognitionPreprocess>  preprocessor;
    std::unique_ptr<RecognitionPostprocess> postprocessor;
};
