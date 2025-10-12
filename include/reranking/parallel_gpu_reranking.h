#ifndef PARALLEL_GPU_RERANKING_H
#define PARALLEL_GPU_RERANKING_H

#include "reranking/neural_reranker.h"
#include "indexing/document.h"
#include "common_types.h"
#include <vector>
#include <string>
#include <unordered_map>
#include <future>

// Forward declarations
class HighPerformanceIRSystem;
class GpuRerankService; // Forward declare the new service class

// Query metrics structure (remains the same)
struct QueryMetrics {
    std::string query_id;
    size_t num_candidates;
    double retrieval_time_ms;
    double reranking_time_ms;
    double precision_at_10;
    double average_precision;
    double reciprocal_rank;
};

// The main parallel processing function, now takes the new service class
std::unordered_map<std::string, std::vector<SearchResult>> 
process_queries_parallel_batched(
    HighPerformanceIRSystem& system,
    GpuRerankService& rerank_service, // Changed to use the new service
    const std::vector<std::pair<std::string, std::string>>& queries,
    const DocumentCollection& documents,
    std::vector<QueryMetrics>& query_metrics
);

#endif