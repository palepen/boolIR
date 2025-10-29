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
    retriever_ = std::make_unique<ParallelRetriever>(index_path, num_shards);
    std::cout << "Using retrieval with " << num_shards << " shards" << std::endl;
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

    std::cout << "\n--- Processed Query Tree ---" << std::endl;
    std::cout << query_str << std::endl;
    std::stringstream ss;
    query_tree->to_string(ss, 0);
    std::cout << ss.str();
    std::cout << "----------------------------" << std::endl;
    ResultSet candidates_result = retriever_->execute_query(*query_tree);

    std::vector<SearchResult> pure_boolean_results;
    pure_boolean_results.reserve(candidates_result.doc_ids.size());

    for (unsigned int doc_id : candidates_result.doc_ids)
    {
        pure_boolean_results.emplace_back(doc_id, 1.0f);
    }

    return pure_boolean_results;
}