#ifndef PARALLEL_GPU_RERANKING_H
#define PARALLEL_GPU_RERANKING_H

#include "reranking/neural_reranker.h"
#include "indexing/document.h"
#include "common_types.h"
#include <vector>
#include <future>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <thread>
#include <string>
#include <unordered_map>

// Forward declarations to avoid circular dependencies
class HighPerformanceIRSystem;

// Query metrics structure (moved from benchmark_suite.h to avoid circular dependency)
struct QueryMetrics {
    std::string query_id;
    size_t num_candidates;
    double retrieval_time_ms;
    double reranking_time_ms;
    double precision_at_10;
    double average_precision;
    double reciprocal_rank;
};

struct QueryBatch {
    std::vector<std::pair<std::string, std::string>> queries;
    std::vector<std::vector<Document>> documents;
    std::vector<std::promise<std::vector<ScoredDocument>>> promises;
};

class BatchedGpuReranker {
private:
    std::unique_ptr<GpuNeuralReranker> reranker_;
    std::queue<QueryBatch> batch_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::atomic<bool> stop_processing_;
    std::thread gpu_worker_;
    
    const size_t MAX_BATCH_SIZE = 4;
    const size_t MAX_WAIT_MS = 10;
    
    void process_batches();
    void process_batch_on_gpu(QueryBatch& batch);
    std::vector<std::string> extract_doc_texts(const std::vector<Document>& docs);
    std::vector<float> extract_query_emb(const std::vector<float>& all_embs, size_t index);

public:
    BatchedGpuReranker(const char* model_path, const char* vocab_path);
    ~BatchedGpuReranker();
    
    std::future<std::vector<ScoredDocument>> submit_query(
        const std::string& query_id,
        const std::string& query_text,
        const std::vector<Document>& docs
    );
};

// Parallel batched query processing function
std::unordered_map<std::string, std::vector<SearchResult>> 
process_queries_parallel_batched(
    HighPerformanceIRSystem& system,
    BatchedGpuReranker& gpu_reranker,
    const std::vector<std::pair<std::string, std::string>>& queries,
    const DocumentCollection& documents,
    std::vector<QueryMetrics>& query_metrics
);

#endif