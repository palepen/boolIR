#include "system_controller.h"
#include "retrieval/query_preprocessor.h"
#include <iostream>
#include <algorithm>
#include <stdexcept>
#include <sstream>
#include <unordered_set>
#include <filesystem>

namespace fs = std::filesystem;

HighPerformanceIRSystem::HighPerformanceIRSystem(
    const std::string &index_path,
    const std::string &synonym_path,
    size_t num_shards)
{
    query_expander_ = std::make_unique<QueryExpander>(synonym_path);
    query_preprocessor_ = std::make_unique<QueryPreprocessor>();
    retriever_ = std::make_unique<DynamicParallelRetriever>(index_path, num_shards);
    std::cout << "Using dynamic sharded retrieval with " << num_shards << " shards" << std::endl;
}

std::unique_ptr<QueryNode> HighPerformanceIRSystem::expand_query(const std::string &query_str)
{
    std::string preprocessed = query_preprocessor_->preprocess(query_str);
    return query_expander_->expand_query(preprocessed);
}

std::vector<SearchResult> HighPerformanceIRSystem::search_boolean(
    const std::string &query_str)
{
    std::unique_ptr<QueryNode> query_tree = expand_query(query_str);

    ResultSet candidates_result = retriever_->execute_query(*query_tree);
    std::sort(candidates_result.doc_ids.begin(), candidates_result.doc_ids.end());

    candidates_result.doc_ids.erase(
        std::unique(candidates_result.doc_ids.begin(), candidates_result.doc_ids.end()),
        candidates_result.doc_ids.end()
    );
    
    std::vector<SearchResult> pure_boolean_results;
    pure_boolean_results.reserve(candidates_result.doc_ids.size());

    for (unsigned int doc_id : candidates_result.doc_ids)
    {
        // Assign a uniform score of 1.0f to all matching documents.
        pure_boolean_results.emplace_back(doc_id, 1.0f);
    }

    return pure_boolean_results;
}