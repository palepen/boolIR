#ifndef BENCHMARK_SUITE_H
#define BENCHMARK_SUITE_H

#include "evaluation/evaluator.h"
#include "data_loader.h"
#include "common_types.h"
#include <vector>
#include <string>
#include <unordered_map>

// Forward declarations
class HighPerformanceIRSystem;
class BatchedGpuReranker;

// Import QueryMetrics from parallel_gpu_reranking.h
struct QueryMetrics;

struct BenchmarkConfig {
    size_t num_workers;
    bool use_reranking;
    bool parallel_queries;
    std::string label;
};

struct BenchmarkResults {
    BenchmarkConfig config;
    double total_time_ms;
    double indexing_time_ms;
    double query_processing_time_ms;
    double throughput_queries_per_sec;
    EvaluationResults effectiveness;
    double avg_retrieval_time_ms;
    double avg_reranking_time_ms;
    double median_retrieval_time_ms;
    double p95_retrieval_time_ms;
    std::vector<QueryMetrics> query_metrics;
};

class BenchmarkSuite {
private:
    const DocumentCollection& documents_;
    const std::unordered_map<std::string, std::string>& topics_;
    const Qrels& ground_truth_;
    const char* model_path_;
    const char* vocab_path_;
    
    void calculate_statistics(BenchmarkResults& results);
    
    std::unordered_map<std::string, std::vector<SearchResult>> 
    process_queries_sequential(
        HighPerformanceIRSystem& system,
        bool use_reranking,
        std::vector<QueryMetrics>& query_metrics
    );

public:
    BenchmarkSuite(
        const DocumentCollection& documents,
        const std::unordered_map<std::string, std::string>& topics,
        const Qrels& ground_truth,
        const char* model_path,
        const char* vocab_path
    );
    
    BenchmarkResults run_benchmark(const BenchmarkConfig& config);
    
    std::vector<BenchmarkResults> run_scalability_test(
        bool use_reranking, 
        const std::vector<size_t>& worker_counts
    );
    
    std::vector<BenchmarkResults> run_comparison_test(size_t num_workers);
    
    void export_to_csv(
        const std::vector<BenchmarkResults>& results,
        const std::string& filename
    );
    
    void export_query_metrics_csv(
        const BenchmarkResults& results,
        const std::string& filename
    );
};

#endif