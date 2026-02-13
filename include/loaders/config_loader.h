#pragma once
#include <iostream>
#include <optional>
#include <string>
#include <yaml-cpp/yaml.h>

#define DET_ONNX_MODEL_FILEPATH "det_onnx_model_filepath"
#define DET_THRESH              "det_thresh"
#define DET_BOX_THRESH          "det_box_thresh"
#define DET_MAX_CANDIDATES      "det_max_candidates"
#define DET_UNCLIP_RATIO        "det_unclip_ratio"
#define DET_USE_DILATION        "det_use_dilation"
#define CLS_ONNX_MODEL_FILEPATH "cls_onnx_model_filepath"
#define CLS_IMAGE_SHAPE         "cls_image_shape"
#define CLS_BATCH_SIZE          "cls_batch_size"
#define CLS_THRESH              "cls_thresh"
#define CLS_LABEL_LIST          "cls_label_list"
#define REC_ONNX_MODEL_FILEPATH "rec_onnx_model_filepath"
#define REC_IMAGE_SHAPE         "rec_image_shape"
#define REC_BATCH_SIZE          "rec_batch_size"
#define REC_CHAR_DICT_PATH      "rec_char_dict_path"
#define REC_USE_SPACE_CHAR      "rec_use_space_char"
#define REC_NORM_MEAN           "rec_norm_mean"
#define REC_NORM_STD            "rec_norm_std"
#define REC_NORM_SCALE          "rec_norm_scale"
#define DROP_SCORE              "drop_score"
#define SIDE_LENGTH_LIMIT       "side_length_limit"
#define LIMIT_TYPE              "limit_type"
#define IMAGE_PATH              "image_path"

class ConfigLoader
{
  public:
    ConfigLoader(const std::string &config_filepath);
    template <typename T> std::optional<T> get(const std::string &key);

  private:
    YAML::Node config_file;

    YAML::Node load_config_file(const std::string &filepath);
};

template <typename T> std::optional<T> ConfigLoader::get(const std::string &key)
{
    if (config_file[key])
    {
        try
        {
            return config_file[key].as<T>();
        }
        catch (const YAML::TypedBadConversion<T> &e)
        {
            std::cerr << "Warning: Key '" << key
                      << "' exists but cannot be converted to the requested "
                         "type. Using default.\n";
        }
    }
    else
    {
        std::cerr << "Warning: Key '" << key << "' not found. Using default.\n";
    }
    return std::nullopt;
}
