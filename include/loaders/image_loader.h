#pragma once
#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>

class ImageLoader
{
  public:
    ImageLoader(const std::string &image_path);
    std::shared_ptr<std::unordered_map<std::string, cv::Mat>> get_images();

  private:
    std::unordered_map<std::string, cv::Mat> images;
    static const std::set<std::string>       supported_extensions;

    void load_images(const std::string &image_filepath);
    void read_image(const std::string &image_filepath);
};
