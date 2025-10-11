#ifndef COMMON_TYPES_H
#define COMMON_TYPES_H

struct SearchResult {
    unsigned int doc_id;
    float score;

    // For sorting results in descending order of score
    bool operator<(const SearchResult& other) const {
        return score > other.score;
    }

    SearchResult(unsigned int doc_id, float score) : doc_id(doc_id), score(score) {}
};

#endif