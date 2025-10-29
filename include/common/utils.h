#ifndef COMMON_UTILS_H
#define COMMON_UTILS_H

#include <string>
#include <sstream>
#include <cstddef>

// Truncates a string to a maximum number of words.
std::string truncate_to_words(const std::string &text, size_t max_words);

struct SearchResult {
    unsigned int doc_id;
    float score;

    bool operator<(const SearchResult& other) const {
        return score > other.score;
    }

    SearchResult() : doc_id(0), score(0.0f) {}
    
    SearchResult(unsigned int doc_id, float score) : doc_id(doc_id), score(score) {}
};


#endif 