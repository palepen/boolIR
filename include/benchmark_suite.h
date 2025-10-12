#ifndef BENCHMARK_SUITE_H
#define BENCHMARK_SUITE_H

#include "evaluation/evaluator.h"
#include "data_loader.h"
#include "common_types.h"
#include "reranking/parallel_gpu_reranking.h" // For QueryMetrics
#include <vector>
#include <string>
#include <unordered_map>
#include <queue>
// Forward declarations
class HighPerformanceIRSystem;

struct BenchmarkConfig {
    size_t num_workers; // Note: For reranking, this is conceptual. For boolean, it's not used.
    bool use_reranking;
    std::string label;
};

struct BenchmarkResults {
    BenchmarkConfig config;
    double total_time_ms;
    double query_processing_time_ms;
    double throughput_queries_per_sec;
    EvaluationResults effectiveness;
    // Latency stats
    double avg_retrieval_time_ms;
    double avg_reranking_time_ms;
    double median_retrieval_time_ms;
    double p95_retrieval_time_ms;
    std::vector<QueryMetrics> query_metrics;
};

struct RerankJob {
    std::string query_text;
    std::vector<Document> candidates;
    std::promise<std::vector<ScoredDocument>> promise;
};

class GpuRerankService {
public:
    GpuRerankService(const std::string& model_path, const std::string& vocab_path);
    ~GpuRerankService();
    std::future<std::vector<ScoredDocument>> submit_job(const std::string& query_text, const std::vector<Document>& candidates);
private:
    void worker_loop();
    GpuNeuralReranker reranker_;
    std::thread worker_thread_;
    std::queue<RerankJob> job_queue_;
    std::mutex queue_mutex_;
    std::condition_variable condition_;
    bool stop_ = false;
};


class BenchmarkSuite {
public:
    BenchmarkSuite(
        const DocumentCollection& documents,
        const std::unordered_map<std::string, std::string>& topics,
        const Qrels& ground_truth,
        const std::string& model_path,
        const std::string& vocab_path,
        const std::string& index_path,
        const std::string& synonym_path
    );
    void run_full_benchmark();
private:

    BenchmarkResults run_benchmark(const BenchmarkConfig& config);
    
    std::vector<BenchmarkResults> run_scalability_test(
        bool use_reranking, 
        const std::vector<size_t>& worker_counts
    );
    
    std::vector<BenchmarkResults> run_comparison_test(size_t num_workers);
    
    void calculate_statistics(BenchmarkResults& results);
    void export_to_csv(const std::vector<BenchmarkResults>& results, const std::string& filename);
    
    // Member variables
    const DocumentCollection& documents_;
    const std::unordered_map<std::string, std::string>& topics_;
    const Qrels& ground_truth_;
    std::string model_path_;
    std::string vocab_path_;
    std::string index_path_;
    std::string synonym_path_;
};

#endif