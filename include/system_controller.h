#pragma once

#include "indexing/parallel_indexer.h"
#include "retrieval/boolean_retrieval.h"
#include "reranking/neural_reranker.h"

// A structure to hold the final search results
struct SearchResult {
    unsigned int doc_id;
    float score;

    // For sorting results in descending order of score
    bool operator<(const SearchResult& other) const {
        return score > other.score;
    }

    SearchResult(unsigned int doc_id, float score) : doc_id(doc_id), score(score) {}
};

class HighPerformanceIRSystem {
public:
    // Constructor initializes all components
    HighPerformanceIRSystem(size_t num_shards, const char* model_path);

    // High-level operations
    void build_index(const DocumentCollection& documents);
    std::vector<SearchResult> search(const std::string& query, bool use_reranking = true);

private:
    // System components
    ParallelIndexer indexer_;
    std::unique_ptr<ParallelBooleanRetrieval> retriever_;
    std::unique_ptr<GpuNeuralReranker> reranker_;

    // Private state
    const DocumentCollection* documents_ptr_ = nullptr; // Pointer to the document collection
};