#pragma once

#include <cstdint>
#include <memory>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <unordered_map>
#include <vector>

#include "ocr_pipeline/detection/postprocess/detection_postprocess.h"
#include "ocr_pipeline/detection/preprocess/detection_preprocess.h"
#include "onnx_predictor/onnx_model_info.h"
#include "utils/utils.h"

class Detector
{
  public:
    Detector(Ort::Env &env, const Ort::SessionOptions &session_options,
             Ort::MemoryInfo &memory_info, const std::string &model_filepath,
             std::unordered_map<std::string, OnnxModelInputInfo> &models,
             bool keep_ratio, int side_length_limit,
             const std::string &limit_type, float threshold = 0.3f,
             float box_threshold = 0.6f, int max_candidates = 1000,
             float unclip_ratio = 1.5f, bool use_dilation = false);

    const std::string &get_limit_type() const;
    void               set_limit_type(std::string limit_type);
    std::vector<Box>
    run(const std::pair<const std::string, std::shared_ptr<cv::Mat>> &image);

  private:
    struct ImageDimensions
    {
        int64_t channels;
        int64_t height;
        int64_t width;
    };

    Ort::Env                  &env;
    const Ort::SessionOptions &session_options;
    Ort::MemoryInfo           &memory_info;

    std::string                                          model_filepath;
    std::unordered_map<std::string, OnnxModelInputInfo> &models;

    bool        keep_ratio;
    int         side_length_limit;
    std::string limit_type;

    std::unique_ptr<DetectionPostprocessor> postprocessor;

    OnnxModelInputInfo *get_model_info() const;

    std::unique_ptr<DetectionPreprocessor> create_preprocessor(
        const OnnxModelInputInfo &model_info,
        const std::pair<const std::string, std::shared_ptr<cv::Mat>> &image)
        const;

    ImageDimensions extract_image_dims(const cv::Mat &image) const;

    void update_image_shape(OnnxModelInputInfo &model_info,
                            int64_t image_height, int64_t image_width) const;

    std::vector<int64_t> build_input_shape(const OnnxModelInputInfo &model_info,
                                           int64_t image_channels,
                                           int64_t image_height,
                                           int64_t image_width) const;

    Ort::Value create_input_tensor(
        const cv::Mat &image, const std::vector<int64_t> &input_shape) const;

    Ort::Value prepare_input_tensor(const cv::Mat            &image,
                                    const OnnxModelInputInfo &model_info) const;

    std::unique_ptr<Ort::Session> create_session() const;

    std::vector<const char *> get_output_names(
        Ort::Session                         &session,
        std::vector<Ort::AllocatedStringPtr> &output_name_allocs) const;

    std::vector<Ort::Value> run_inference(
        Ort::Session &session, const std::vector<const char *> &input_names,
        Ort::Value                      &input_tensor,
        const std::vector<const char *> &output_names) const;

    cv::Mat build_prediction_maps(Ort::Value &output_tensor) const;
};
