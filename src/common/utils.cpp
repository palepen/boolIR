#include "common/utils.h"
#include <sstream>

std::string truncate_to_words(const std::string &text, size_t max_words)
{
    std::istringstream iss(text);
    std::ostringstream oss;
    std::string word;
    for (size_t count = 0; count < max_words && iss >> word; ++count)
    {
        oss << (count > 0 ? " " : "") << word;
    }
    return oss.str();
}