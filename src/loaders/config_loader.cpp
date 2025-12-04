#include "loaders/config_loader.h"

ConfigLoader::ConfigLoader(const std::string &config_filepath)
{
    config_file = loadConfigFile(config_filepath);
}

YAML::Node ConfigLoader::loadConfigFile(const std::string &filepath)
{
    try
    {
        return YAML::LoadFile(filepath);
    }
    catch (const YAML::BadFile &e)
    {
        std::cerr << "Error: Could not open config file '" << filepath
                  << "'.\n";
        throw std::runtime_error("Could not load the config file!");
    }
    catch (const YAML::ParserException &e)
    {
        std::cerr << "Error: Failed to parse config file '" << filepath
                  << "': " << e.what() << std::endl;
        throw std::runtime_error("Could not load the config file!");
    }
}
