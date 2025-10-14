#ifndef BENCHMARK_SUITE_H
#define BENCHMARK_SUITE_H

#include "evaluation/evaluator.h"
#include "data_loader.h"
#include "common_types.h"
#include <vector>
#include <string>
#include <unordered_map>
#include <queue>
#include <mutex>
#include <future>

// Forward declarations
class HighPerformanceIRSystem;

struct BenchmarkConfig {
    size_t num_cpu_workers;
    bool use_reranking;
    std::string label;
};

struct QueryMetrics {
    std::string query_id;
    size_t num_candidates;
    double retrieval_time_ms = 0.0;
    double reranking_time_ms = 0.0;
};

struct BenchmarkResults {
    BenchmarkConfig config;
    double query_processing_time_ms;
    double throughput_queries_per_sec;
    EvaluationResults effectiveness;
    // Latency stats
    double avg_retrieval_time_ms;
    double avg_reranking_time_ms;
    double median_latency_ms;
    double p95_latency_ms;
    std::vector<QueryMetrics> query_metrics;
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
    
    void run_single_benchmark(const BenchmarkConfig& config);
    void print_comparison() const;

private:
    void calculate_statistics(BenchmarkResults& results);
    void export_to_csv(const BenchmarkResults& result, const std::string& filename);
    
    // Member variables
    const DocumentCollection& documents_;
    const std::unordered_map<std::string, std::string>& topics_;
    const Qrels& ground_truth_;
    std::string model_path_;
    std::string vocab_path_;
    std::string index_path_;
    std::string synonym_path_;
    
    // Store last results for comparison
    BenchmarkResults last_boolean_results_;
    BenchmarkResults last_reranking_results_;
};

#endif