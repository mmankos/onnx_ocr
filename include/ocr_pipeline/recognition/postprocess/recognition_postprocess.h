#pragma once

#include <algorithm>
#include <fstream>
#include <regex>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

class RecognitionPostprocess
{
  public:
    RecognitionPostprocess(const std::string &character_dict_path,
                           bool               use_space_char);

    std::vector<std::pair<std::string, float>> decode(const float *predictions,
                                                      size_t       batch_size,
                                                      size_t       time_steps,
                                                      size_t num_classes) const;
    size_t                                     num_classes() const;

  private:
    std::string prediction_reverse(const std::string &prediction) const;
    void        load_characters(const std::string &character_dict_path,
                                bool               use_space_char);

    std::vector<std::string> characters;
    bool                     reverse{false};
};
