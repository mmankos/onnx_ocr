#pragma once

#include <string>
#include <utility>
#include <vector>

class DirectionalClassificationPostprocess
{
  public:
    explicit DirectionalClassificationPostprocess(
        std::vector<std::string> label_list);

    std::vector<std::pair<std::string, float>> decode(const float *predictions,
                                                      size_t       batch_size,
                                                      size_t num_classes) const;

  private:
    std::vector<std::string> label_list;
};
