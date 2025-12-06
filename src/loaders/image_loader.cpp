#include "loaders/image_loader.h"

const std::set<std::string> ImageLoader::supported_extensions = {
    ".bmp",  ".dib",  ".jpeg", ".jpg", ".jpe", ".jp2", ".png", ".webp",
    ".avif", ".pbm",  ".pgm",  ".ppm", ".pxm", ".pnm", ".pfm", ".sr",
    ".ras",  ".tiff", ".tif",  ".exr", ".hdr", ".pic"};

ImageLoader::ImageLoader(const std::string &image_path)
{
    load_images(image_path);
}

std::shared_ptr<std::unordered_map<std::string, cv::Mat>>
ImageLoader::get_images()
{
    return std::make_shared<std::unordered_map<std::string, cv::Mat>>(images);
}

void ImageLoader::load_images(const std::string &image_path)
{
    if (!std::filesystem::exists(image_path))
    {
        std::cerr << "[ERROR][ImageLoader::load_images] Path: " << image_path
                  << " does not exist" << std::endl;
        throw std::runtime_error("Invalid path");
    }
    else if (std::filesystem::is_directory(image_path))
    {
        for (const auto &entry :
             std::filesystem::recursive_directory_iterator(image_path))
        {
            std::string entry_extension = entry.path().extension().c_str();
            if (supported_extensions.count(entry_extension))
            {
                std::string entry_path = entry.path().c_str();
                read_image(entry_path);
            }
        }
    }
    else if (std::filesystem::is_regular_file(image_path))
    {
        std::filesystem::path entry_path(image_path);
        std::string           entry_extension = entry_path.extension().c_str();
        if (supported_extensions.count(entry_extension))
        {
            read_image(entry_path);
        }
    }
    else
    {
        std::cerr << "[ERROR][ImageLoader::load_images] Path: " << image_path
                  << " exists but is neither a regular file nor a directory"
                  << std::endl;
        throw std::runtime_error("Invalid path");
    }
}

void ImageLoader::read_image(const std::string &image_filepath)
{
    cv::Mat image = cv::imread(image_filepath, cv::IMREAD_COLOR);
    if (image.empty())
    {
        std::cerr << "[ERROR][ImageLoader::read_image] Failed to read image: "
                  << image_filepath << std::endl;
    }
    else
    {
        cv::Mat img_rgb;
        int     image_channels = image.channels();

        if (image_channels == 1)
        {
            cv::cvtColor(image, img_rgb, cv::COLOR_GRAY2RGB);
        }
        else if (image_channels == 3)
        {
            cv::cvtColor(image, img_rgb, cv::COLOR_BGR2RGB);
        }
        else if (image_channels == 4)
        {
            cv::cvtColor(image, img_rgb, cv::COLOR_BGRA2RGB);
        }
        else
        {
            std::cerr << "[ERROR][ImageLoader::read_image] Image has an "
                         "unsupported number of channels: "
                      << image.channels() << std::endl;
            return;
        }
        images[image_filepath] = img_rgb;
    }
}
