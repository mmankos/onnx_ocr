#include "ocr_pipeline/recognition/postprocess/recognition_postprocess.h"

RecognitionPostprocess::RecognitionPostprocess(
    const std::string &character_dict_path, bool use_space_char)
{
    load_characters(character_dict_path, use_space_char);
}

void RecognitionPostprocess::load_characters(
    const std::string &character_dict_path, bool use_space_char)
{
    std::vector<std::string> dict_character;
    if (character_dict_path.empty())
    {
        const std::string default_chars =
            "0123456789abcdefghijklmnopqrstuvwxyz";
        for (char c : default_chars) { dict_character.emplace_back(1, c); }
    }
    else
    {
        std::ifstream fin(character_dict_path);
        std::string   line;
        while (std::getline(fin, line))
        {
            if (!line.empty() && line.back() == '\r')
            {
                line.pop_back();
            }
            if (!line.empty())
            {
                dict_character.push_back(line);
            }
        }

        if (character_dict_path.find("arabic") != std::string::npos)
        {
            reverse = true;
        }
    }

    if (use_space_char)
    {
        dict_character.emplace_back(" ");
    }

    characters.clear();
    characters.emplace_back("blank");
    characters.insert(characters.end(), dict_character.begin(),
                      dict_character.end());
}

std::string
RecognitionPostprocess::prediction_reverse(const std::string &prediction) const
{
    std::vector<std::string> groups;
    std::string              current;
    std::regex               alnum_re("[a-zA-Z0-9 :*./%+-]");

    for (char c : prediction)
    {
        std::string s(1, c);
        if (!std::regex_match(s, alnum_re))
        {
            if (!current.empty())
            {
                groups.push_back(current);
                current.clear();
            }
            groups.emplace_back(s);
        }
        else
        {
            current += c;
        }
    }
    if (!current.empty())
    {
        groups.push_back(current);
    }

    std::ostringstream oss;
    for (auto it = groups.rbegin(); it != groups.rend(); ++it) { oss << *it; }
    return oss.str();
}

std::vector<std::pair<std::string, float>>
RecognitionPostprocess::decode(const float *predictions, size_t batch_size,
                               size_t time_steps, size_t num_classes) const
{
    std::vector<std::pair<std::string, float>> results;
    results.reserve(batch_size);

    for (size_t b = 0; b < batch_size; ++b)
    {
        std::string text;
        float       conf_sum = 0.0f;
        size_t      conf_cnt = 0;
        int         prev_idx = -1;

        const float *batch_ptr = predictions + b * time_steps * num_classes;
        for (size_t t = 0; t < time_steps; ++t)
        {
            const float *row      = batch_ptr + t * num_classes;
            size_t       best_idx = 0;
            float        best_val = row[0];
            for (size_t c = 1; c < num_classes; ++c)
            {
                if (row[c] > best_val)
                {
                    best_val = row[c];
                    best_idx = c;
                }
            }

            if (best_idx == 0 || static_cast<int>(best_idx) == prev_idx)
            {
                prev_idx = static_cast<int>(best_idx);
                continue;
            }

            prev_idx = static_cast<int>(best_idx);
            if (best_idx < characters.size())
            {
                text += characters[best_idx];
                conf_sum += best_val;
                conf_cnt += 1;
            }
        }

        float score = conf_cnt > 0 ? conf_sum / conf_cnt : 0.0f;
        if (reverse)
        {
            text = prediction_reverse(text);
        }

        results.emplace_back(text, score);
    }

    return results;
}

size_t RecognitionPostprocess::num_classes() const { return characters.size(); }
