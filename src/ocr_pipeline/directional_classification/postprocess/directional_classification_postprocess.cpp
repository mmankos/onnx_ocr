#include "ocr_pipeline/directional_classification/postprocess/directional_classification_postprocess.h"

DirectionalClassificationPostprocess::DirectionalClassificationPostprocess(
    std::vector<std::string> label_list)
    : label_list(std::move(label_list))
{}

std::vector<std::pair<std::string, float>>
DirectionalClassificationPostprocess::decode(const float *predictions,
                                             size_t       batch_size,
                                             size_t       num_classes) const
{
    std::vector<std::pair<std::string, float>> results;
    results.reserve(batch_size);

    for (size_t i = 0; i < batch_size; ++i)
    {
        const float *row        = predictions + i * num_classes;
        size_t       best_idx   = 0;
        float        best_score = row[0];
        for (size_t c = 1; c < num_classes; ++c)
        {
            if (row[c] > best_score)
            {
                best_score = row[c];
                best_idx   = c;
            }
        }

        if (best_idx < label_list.size())
        {
            results.emplace_back(label_list[best_idx], best_score);
        }
        else
        {
            results.emplace_back(std::to_string(best_idx), best_score);
        }
    }

    return results;
}
