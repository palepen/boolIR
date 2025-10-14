#ifndef PRE_RANKER_H
#define PRE_RANKER_H

#include "retrieval/result_set.h"
#include "indexing/document.h"
#include "common_types.h"
#include <vector>
#include <string>
#include <unordered_map>

class PreRanker {
public:
    virtual ~PreRanker() = default;

    virtual std::vector<SearchResult> rank(
        const std::string& original_query,
        const ResultSet& candidates,
        const std::unordered_map<unsigned int, const Document*>& doc_id_map  // CONSISTENT NAME
    ) const = 0;
};

class TermOverlapRanker : public PreRanker {
public:
    std::vector<SearchResult> rank(
        const std::string& original_query,
        const ResultSet& candidates,
        const std::unordered_map<unsigned int, const Document*>& doc_id_map  // CONSISTENT NAME
    ) const override;
};

#endif // PRE_RANKER_H
