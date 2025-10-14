#ifndef COMMON_TYPES_H
#define COMMON_TYPES_H

struct SearchResult {
    unsigned int doc_id;
    float score;

    // For sorting results in descending order of score
    bool operator<(const SearchResult& other) const {
        return score > other.score;
    }

    // Default constructor (needed for vector resize)
    SearchResult() : doc_id(0), score(0.0f) {}
    
    SearchResult(unsigned int doc_id, float score) : doc_id(doc_id), score(score) {}
};

#endif