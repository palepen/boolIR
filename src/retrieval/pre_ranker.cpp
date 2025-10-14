#include "retrieval/pre_ranker.h"
#include <sstream>
#include <unordered_set>
#include <algorithm>
#include <cctype>

// Helper function to tokenize text for overlap calculation.
// This is kept internal to the implementation file.
static std::unordered_set<std::string> get_unique_terms(const std::string& text) {
    std::unordered_set<std::string> terms;
    std::stringstream ss(text);
    std::string token;
    while (ss >> token) {
        // Remove punctuation
        token.erase(std::remove_if(token.begin(), token.end(),
            [](unsigned char c) { return !std::isalnum(c); }), token.end());

        if (!token.empty()) {
            // Convert to lowercase
            std::transform(token.begin(), token.end(), token.begin(),
                           [](unsigned char c){ return std::tolower(c); });
            terms.insert(token);
        }
    }
    return terms;
}

std::vector<SearchResult> TermOverlapRanker::rank(
    const std::string& original_query,
    const ResultSet& candidates,
    const std::unordered_map<unsigned int, const Document*>& doc_id_map
) const {
    // 1. Get the unique terms from the original user query.
    std::unordered_set<std::string> query_terms = get_unique_terms(original_query);
    if (query_terms.empty()) {
        return {};
    }

    std::vector<SearchResult> results;
    results.reserve(candidates.doc_ids.size());

    for (unsigned int doc_id : candidates.doc_ids) {
        auto it = doc_id_map.find(doc_id);
        if (it != doc_id_map.end()) {
            // 2. Get unique terms from the document content.
            std::unordered_set<std::string> doc_terms = get_unique_terms(it->second->content);

            // 3. Calculate the overlap score.
            float overlap_count = 0.0f;
            for (const auto& term : query_terms) {
                if (doc_terms.count(term)) {
                    overlap_count++;
                }
            }

            float score = overlap_count;
            results.push_back({doc_id, score});
        }
    }

    // 4. Sort the results so documents with the highest overlap score are first.
    std::sort(results.begin(), results.end());

    return results;
}

