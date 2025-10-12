#ifndef SYSTEM_CONTROLLER_H
#define SYSTEM_CONTROLLER_H

#include "retrieval/optimized_parallel_retrieval.h"
#include "retrieval/query_expander.h"
#include "reranking/neural_reranker.h"
#include "common_types.h"
#include <memory>
#include <vector>
#include <string>

/**
 * High-Performance IR System
 * Pure Boolean retrieval with simple heuristic ordering + Neural reranking
 */
class HighPerformanceIRSystem {
public:
    HighPerformanceIRSystem(const std::string& index_path, const std::string& synonym_path);
    
    // Search with neural reranking
    std::vector<SearchResult> search(
        const std::string& query_str, 
        GpuNeuralReranker& reranker,
        const DocumentCollection& documents
    );

    // Search with Boolean only (+ simple heuristics)
    std::vector<SearchResult> search(
        const std::string& query_str,
        const DocumentCollection& documents
    );
    
    ResultSet execute_boolean_query(const QueryNode& query);

    std::unique_ptr<QueryNode> expand_query(const std::string& query_str);

private:
    void ensure_doc_map(const DocumentCollection& documents);
    
    // Simple heuristic ordering for Boolean results
    std::vector<SearchResult> apply_simple_heuristics(
        const ResultSet& candidates,
        const DocumentCollection& documents
    );

    std::unique_ptr<QueryExpander> query_expander_;
    std::unique_ptr<OptimizedParallelRetrieval> retriever_;
    std::unordered_map<unsigned int, const Document*> doc_id_map_;
};

#endif