#ifndef SYSTEM_CONTROLLER_H
#define SYSTEM_CONTROLLER_H

#include "indexing/parallel_indexer.h"
#include "retrieval/optimized_parallel_retrieval.h"
#include "reranking/parallel_gpu_reranking.h"
#include <memory>
#include <vector>
#include <string>
#include <unordered_map>
#include <chrono>

/**
 * High-Performance Information Retrieval System
 * 
 * Integrates:
 * 1. Parallel indexing (OpenCilk)
 * 2. Optimized boolean retrieval (fast candidate selection)
 * 3. Neural reranking (BERT-based semantic relevance scoring)
 */
class HighPerformanceIRSystem {
public:
    /**
     * Constructor
     * @param num_shards Number of index shards for parallel processing
     * @param model_path Path to ONNX model file
     * @param vocab_path Path to BERT vocabulary file
     */
    HighPerformanceIRSystem(size_t num_shards, const char* model_path, const char* vocab_path);
    
    /**
     * Build inverted index from document collection
     * Also builds document ID mapping for fast lookup
     * @param documents Collection of documents to index
     */
    void build_index(const DocumentCollection& documents);
    
    /**
     * Search for documents matching query
     * @param query_str Query string
     * @param use_reranking Whether to apply neural reranking (default: true)
     * @return Ranked list of search results
     */
    std::vector<SearchResult> search(const std::string& query_str, bool use_reranking = true);
    
    /**
     * Execute boolean query directly (for benchmarking)
     * @param query QueryNode tree
     * @return ResultSet with matching document IDs
     */
    ResultSet execute_boolean_query(const QueryNode& query);

private:
    // Core components
    ParallelIndexer indexer_;
    std::unique_ptr<ParallelRetrieval> retriever_;
    std::unique_ptr<BatchedGpuReranker> reranker_;
    
    // Document storage and mapping
    const DocumentCollection* documents_ptr_ = nullptr;
    
    // CRITICAL: Document ID to Document pointer mapping for O(1) lookup
    // This replaces the inefficient O(n) linear search through all documents
    std::unordered_map<unsigned int, const Document*> doc_id_map_;
};

#endif 