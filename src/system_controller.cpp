#include "system_controller.h"
#include "retrieval/query_preprocessor.h"
#include <iostream>
#include <algorithm>
#include <stdexcept>
#include <sstream>
#include <unordered_set>

HighPerformanceIRSystem::HighPerformanceIRSystem(
    const std::string& index_path, 
    const std::string& synonym_path,
    std::unique_ptr<PreRanker> pre_ranker
) : pre_ranker_(std::move(pre_ranker)) {
    query_expander_ = std::make_unique<QueryExpander>(synonym_path);
    query_preprocessor_ = std::make_unique<QueryPreprocessor>();
    
    std::string dict_path = index_path + "/dictionary.dat";
    std::string post_path = index_path + "/postings.dat";
    retriever_ = std::make_unique<OptimizedParallelRetrieval>(dict_path, post_path);
}

std::unique_ptr<QueryNode> HighPerformanceIRSystem::expand_query(const std::string& query_str) {
    if (!query_expander_) {
        throw std::runtime_error("Query expander is not initialized.");
    }
    
    // First preprocess the query for consistency
    if (!query_preprocessor_) {
        throw std::runtime_error("Query preprocessor is not initialized.");
    }
    
    std::string preprocessed = query_preprocessor_->preprocess(query_str);
    
    // Then expand with synonyms
    return query_expander_->expand_query(preprocessed);
}

std::vector<SearchResult> HighPerformanceIRSystem::search_boolean(
    const std::string& query_str,
    const DocumentCollection& documents
) {
    std::unique_ptr<QueryNode> query_tree = expand_query(query_str);
    ResultSet candidates_result = retriever_->execute_query_optimized(*query_tree);

    // Create the map for document lookup
    std::unordered_map<unsigned int, const Document*> doc_id_map;
    doc_id_map.reserve(documents.size());
    for(const auto& doc : documents) {
        doc_id_map[doc.id] = &doc;
    }

    if (!pre_ranker_) {
        throw std::runtime_error("Pre-ranker is not initialized.");
    }
    
    // Use the original query (before preprocessing) for ranking
    // This preserves the user's intent for term overlap scoring
    return pre_ranker_->rank(query_str, candidates_result, doc_id_map);
}