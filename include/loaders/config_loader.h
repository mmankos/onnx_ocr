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
#define REC_ONNX_MODEL_FILEPATH "rec_onnx_model_filepath"
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
