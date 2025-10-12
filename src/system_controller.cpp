#include "system_controller.h"
#include <iostream>
#include <algorithm>
#include <stdexcept>
#include <random>

HighPerformanceIRSystem::HighPerformanceIRSystem(const std::string& index_path, const std::string& synonym_path) {
    query_expander_ = std::make_unique<QueryExpander>(synonym_path);
    
    std::string dict_path = index_path + "/dictionary.dat";
    std::string post_path = index_path + "/postings.dat";
    retriever_ = std::make_unique<OptimizedParallelRetrieval>(dict_path, post_path);
}

std::unique_ptr<QueryNode> HighPerformanceIRSystem::expand_query(const std::string& query_str) {
    if (!query_expander_) {
        throw std::runtime_error("Query expander is not initialized.");
    }
    return query_expander_->expand_query(query_str);
}

void HighPerformanceIRSystem::ensure_doc_map(const DocumentCollection& documents) {
    if (doc_id_map_.empty()) {
        for(const auto& doc : documents) {
            doc_id_map_[doc.id] = &doc;
        }
    }
}

/**
 * Simple heuristic for ordering Boolean results:
 * Prefer shorter documents (more focused content)
 */
std::vector<SearchResult> HighPerformanceIRSystem::apply_simple_heuristics(
    const ResultSet& candidates,
    const DocumentCollection& documents
) {
    ensure_doc_map(documents);
    
    std::vector<SearchResult> results;
    results.reserve(candidates.doc_ids.size());
    
    for (unsigned int doc_id : candidates.doc_ids) {
        auto it = doc_id_map_.find(doc_id);
        if (it != doc_id_map_.end()) {
            // Simple heuristic: shorter documents get higher "pseudo-score"
            // This helps surface more focused, relevant content
            size_t doc_length = it->second->content.length();
            float pseudo_score = 1.0f / (1.0f + doc_length / 1000.0f);
            results.push_back({doc_id, pseudo_score});
        }
    }
    
    // Sort by pseudo-score (descending)
    std::sort(results.begin(), results.end());
    
    return results;
}

std::vector<SearchResult> HighPerformanceIRSystem::search(
    const std::string& query_str, 
    GpuNeuralReranker& reranker,
    const DocumentCollection& documents
) {
    ensure_doc_map(documents);

    // Step 1: Boolean retrieval
    std::unique_ptr<QueryNode> query_tree = expand_query(query_str);
    ResultSet candidates_result = retriever_->execute_query_optimized(*query_tree);

    // Step 2: Apply simple heuristics for initial ordering
    auto ordered_results = apply_simple_heuristics(candidates_result, documents);
    
    // Step 3: Take top candidates for reranking
    const size_t max_candidates_to_rerank = 1000;
    std::vector<Document> candidate_docs;
    size_t limit = std::min(max_candidates_to_rerank, ordered_results.size());
    candidate_docs.reserve(limit);

    for (size_t i = 0; i < limit; ++i) {
        unsigned int doc_id = ordered_results[i].doc_id;
        auto it = doc_id_map_.find(doc_id);
        if (it != doc_id_map_.end()) {
            candidate_docs.push_back(*(it->second));
        }
    }

    if (candidate_docs.empty()) {
        return {};
    }

    // Step 4: Neural reranking
    auto reranked_docs = reranker.rerank(query_str, candidate_docs);

    std::vector<SearchResult> final_results;
    final_results.reserve(reranked_docs.size());
    for (const auto &scored_doc : reranked_docs) {
        final_results.push_back({scored_doc.id, scored_doc.score});
    }

    return final_results;
}

std::vector<SearchResult> HighPerformanceIRSystem::search(
    const std::string& query_str,
    const DocumentCollection& documents
) {
    // Pure Boolean retrieval with simple heuristic ordering
    std::unique_ptr<QueryNode> query_tree = expand_query(query_str);
    ResultSet candidates_result = retriever_->execute_query_optimized(*query_tree);

    return apply_simple_heuristics(candidates_result, documents);
}

ResultSet HighPerformanceIRSystem::execute_boolean_query(const QueryNode& query) {
    if (!retriever_) {
        throw std::runtime_error("Retrieval engine not initialized.");
    }
    return retriever_->execute_query_optimized(query);
}