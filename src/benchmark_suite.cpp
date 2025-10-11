#include "benchmark_suite.h"
#include "system_controller.h"
#include "reranking/parallel_gpu_reranking.h"
#include <cilk/cilk.h>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <numeric>
#include <mutex>
#include <atomic>
#include <fstream>

BenchmarkSuite::BenchmarkSuite(
    const DocumentCollection& documents,
    const std::unordered_map<std::string, std::string>& topics,
    const Qrels& ground_truth,
    const char* model_path,
    const char* vocab_path
) : documents_(documents),
    topics_(topics),
    ground_truth_(ground_truth),
    model_path_(model_path),
    vocab_path_(vocab_path) {
}

BenchmarkResults BenchmarkSuite::run_benchmark(const BenchmarkConfig& config) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Running Benchmark: " << config.label << std::endl;
    std::cout << "  Workers: " << config.num_workers << std::endl;
    std::cout << "  Reranking: " << (config.use_reranking ? "Enabled" : "Disabled") << std::endl;
    std::cout << "  Parallel Queries: " << (config.parallel_queries ? "Yes" : "No") << std::endl;
    std::cout << "========================================" << std::endl;
    
    BenchmarkResults results;
    results.config = config;
    
    auto start_total = std::chrono::high_resolution_clock::now();
    
    // Initialize system
    std::cout << "Initializing system..." << std::endl;
    HighPerformanceIRSystem system(config.num_workers, model_path_, vocab_path_);
    
    // Build index
    std::cout << "Building index with " << config.num_workers << " shards..." << std::endl;
    auto start_index = std::chrono::high_resolution_clock::now();
    system.build_index(documents_);
    auto end_index = std::chrono::high_resolution_clock::now();
    results.indexing_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_index - start_index).count();
    
    std::cout << "Index built in " << results.indexing_time_ms << " ms" << std::endl;
    
    // Process queries
    std::cout << "Processing " << topics_.size() << " queries..." << std::endl;
    auto start_queries = std::chrono::high_resolution_clock::now();
    
    std::unordered_map<std::string, std::vector<SearchResult>> search_results;
    
    // CRITICAL FIX: Only use batched GPU path when reranking is enabled
    if (config.parallel_queries && config.use_reranking) {
        // Prepare queries
        std::vector<std::pair<std::string, std::string>> queries;
        queries.reserve(topics_.size());
        for (const auto& [qid, qtext] : topics_) {
            queries.emplace_back(qid, qtext);
        }
        // Initialize batched reranker for parallel processing
        BatchedGpuReranker gpu_reranker(model_path_, vocab_path_);
        search_results = process_queries_parallel_batched(
            system, gpu_reranker, queries, documents_, results.query_metrics
        );
    } else {
        // Sequential or boolean-only processing
        search_results = process_queries_sequential(
            system, 
            config.use_reranking, 
            results.query_metrics
        );
    }
    
    auto end_queries = std::chrono::high_resolution_clock::now();
    results.query_processing_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_queries - start_queries).count();
    
    std::cout << "Query processing completed in " << results.query_processing_time_ms << " ms" << std::endl;
    
    // Evaluate
    std::cout << "Evaluating results..." << std::endl;
    Evaluator evaluator(ground_truth_);
    results.effectiveness = evaluator.evaluate(search_results);
    
    auto end_total = std::chrono::high_resolution_clock::now();
    results.total_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_total - start_total).count();
    
    // Calculate throughput
    results.throughput_queries_per_sec = 
        (1000.0 * topics_.size()) / results.query_processing_time_ms;
    
    // Calculate statistics
    calculate_statistics(results);
    
    // Print summary
    std::cout << "\n┌─────────────────────────────────────┐" << std::endl;
    std::cout << "│      BENCHMARK RESULTS              │" << std::endl;
    std::cout << "├─────────────────────────────────────┤" << std::endl;
    std::cout << "│ Total time:      " << std::setw(10) << results.total_time_ms << " ms  │" << std::endl;
    std::cout << "│ Query processing:" << std::setw(10) << results.query_processing_time_ms << " ms  │" << std::endl;
    std::cout << "│ Throughput:      " << std::fixed << std::setprecision(2) 
              << std::setw(10) << results.throughput_queries_per_sec << " q/s │" << std::endl;
    std::cout << "├─────────────────────────────────────┤" << std::endl;
    std::cout << "│ P@10:            " << std::fixed << std::setprecision(4) 
              << std::setw(10) << results.effectiveness.precision_at_10 << "     │" << std::endl;
    std::cout << "│ MAP:             " << std::setw(10) 
              << results.effectiveness.mean_average_precision << "     │" << std::endl;
    std::cout << "│ MRR:             " << std::setw(10) 
              << results.effectiveness.mean_reciprocal_rank << "     │" << std::endl;
    std::cout << "│ NDCG@10:         " << std::setw(10) 
              << results.effectiveness.ndcg_at_10 << "     │" << std::endl;
    std::cout << "│ DCG@10:          " << std::setw(10) 
              << results.effectiveness.dcg_at_10 << "     │" << std::endl;
    std::cout << "└─────────────────────────────────────┘" << std::endl;
    
    return results;
}

std::unordered_map<std::string, std::vector<SearchResult>> BenchmarkSuite::process_queries_sequential(
    HighPerformanceIRSystem& system, 
    bool use_reranking, 
    std::vector<QueryMetrics>& query_metrics
) {
    std::cout << "  [SEQUENTIAL] Processing queries " 
              << (use_reranking ? "with" : "without") << " reranking..." << std::endl;
    
    std::unordered_map<std::string, std::vector<SearchResult>> results;
    query_metrics.reserve(topics_.size());
    
    size_t completed = 0;
    for (const auto& [qid, qtext] : topics_) {
        QueryMetrics qm;
        qm.query_id = qid;
        
        auto start_retrieval = std::chrono::high_resolution_clock::now();
        auto search_results = system.search(qtext, use_reranking);
        auto end_retrieval = std::chrono::high_resolution_clock::now();
        
        qm.retrieval_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_retrieval - start_retrieval).count();
        qm.reranking_time_ms = use_reranking ? qm.retrieval_time_ms : 0;
        qm.num_candidates = search_results.size();
        
        results[qid] = search_results;
        query_metrics.push_back(qm);
        
        completed++;
        if (completed % 10 == 0 || completed == topics_.size()) {
            std::cout << "  Progress: " << completed << "/" << topics_.size() << std::endl;
        }
    }
    
    return results;
}

void BenchmarkSuite::calculate_statistics(BenchmarkResults& results) {
    if (results.query_metrics.empty()) return;
    
    std::vector<double> retrieval_times;
    retrieval_times.reserve(results.query_metrics.size());
    double sum_retrieval = 0.0;
    double sum_reranking = 0.0;
    
    for (const auto& qm : results.query_metrics) {
        sum_retrieval += qm.retrieval_time_ms;
        sum_reranking += qm.reranking_time_ms;
        retrieval_times.push_back(qm.retrieval_time_ms);
    }
    
    size_t n = results.query_metrics.size();
    results.avg_retrieval_time_ms = sum_retrieval / n;
    results.avg_reranking_time_ms = sum_reranking / n;
    
    std::sort(retrieval_times.begin(), retrieval_times.end());
    results.median_retrieval_time_ms = retrieval_times[n / 2];
    results.p95_retrieval_time_ms = retrieval_times[static_cast<size_t>(n * 0.95)];
}

std::vector<BenchmarkResults> BenchmarkSuite::run_scalability_test(
    bool use_reranking,
    const std::vector<size_t>& worker_counts
) {
    std::vector<BenchmarkResults> results;
    for (size_t num_workers : worker_counts) {
        BenchmarkConfig config;
        config.num_workers = num_workers;
        config.use_reranking = use_reranking;
        config.parallel_queries = use_reranking;  // Only parallel when reranking
        config.label = (use_reranking ? "Reranking_" : "Boolean_") + std::to_string(num_workers) + "w";
        results.push_back(run_benchmark(config));
    }
    return results;
}

std::vector<BenchmarkResults> BenchmarkSuite::run_comparison_test(size_t num_workers) {
    std::vector<BenchmarkResults> results;
    
    // Boolean retrieval only (sequential)
    BenchmarkConfig config_boolean;
    config_boolean.num_workers = num_workers;
    config_boolean.use_reranking = false;
    config_boolean.parallel_queries = false;
    config_boolean.label = "Boolean_" + std::to_string(num_workers) + "w";
    
    auto results_boolean = run_benchmark(config_boolean);
    
    // With neural reranking (parallel)
    BenchmarkConfig config_rerank;
    config_rerank.num_workers = num_workers;
    config_rerank.use_reranking = true;
    config_rerank.parallel_queries = true;
    config_rerank.label = "Reranking_" + std::to_string(num_workers) + "w";
    
    auto results_rerank = run_benchmark(config_rerank);
    
    results.push_back(results_boolean);
    results.push_back(results_rerank);
    
    // Print comparison
    std::cout << "\n╔════════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║                   COMPARISON SUMMARY                           ║" << std::endl;
    std::cout << "╚════════════════════════════════════════════════════════════════╝" << std::endl;
    
    std::cout << "\nMetric                  | Boolean    | Reranking  | Improvement" << std::endl;
    std::cout << "------------------------|------------|------------|-------------" << std::endl;
    
    auto print_comparison_row = [](const std::string& metric, double bool_val, double rerank_val) {
        double improvement = ((rerank_val - bool_val) / (bool_val + 1e-10)) * 100.0;
        std::cout << std::setw(23) << std::left << metric << " | "
                  << std::setw(10) << std::fixed << std::setprecision(4) << bool_val << " | "
                  << std::setw(10) << rerank_val << " | "
                  << std::showpos << std::setw(10) << std::setprecision(1) << improvement << "%" 
                  << std::noshowpos << std::endl;
    };
    
    print_comparison_row("Throughput (q/s)", 
        results_boolean.throughput_queries_per_sec,
        results_rerank.throughput_queries_per_sec);
    
    print_comparison_row("Precision@10", 
        results_boolean.effectiveness.precision_at_10,
        results_rerank.effectiveness.precision_at_10);
    
    print_comparison_row("MAP", 
        results_boolean.effectiveness.mean_average_precision,
        results_rerank.effectiveness.mean_average_precision);
    
    print_comparison_row("MRR", 
        results_boolean.effectiveness.mean_reciprocal_rank,
        results_rerank.effectiveness.mean_reciprocal_rank);
    
    return results;
}

void BenchmarkSuite::export_to_csv(
    const std::vector<BenchmarkResults>& results,
    const std::string& filename
) {
    std::ofstream ofs(filename);
    if (!ofs) {
        std::cerr << "Error: Cannot open " << filename << " for writing" << std::endl;
        return;
    }
    
    // CSV header
    ofs << "label,num_workers,use_reranking,parallel_queries,"
        << "total_time_ms,indexing_time_ms,query_processing_time_ms,"
        << "throughput_qps,precision_at_10,map,mrr,ndcg_at_10,dcg_at_10,"
        << "avg_retrieval_ms,avg_reranking_ms,median_retrieval_ms,p95_retrieval_ms\n";
    
    // Data rows
    for (const auto& r : results) {
        ofs << r.config.label << ","
            << r.config.num_workers << ","
            << (r.config.use_reranking ? 1 : 0) << ","
            << (r.config.parallel_queries ? 1 : 0) << ","
            << r.total_time_ms << ","
            << r.indexing_time_ms << ","
            << r.query_processing_time_ms << ","
            << std::fixed << std::setprecision(2) << r.throughput_queries_per_sec << ","
            << std::setprecision(4) << r.effectiveness.precision_at_10 << ","
            << r.effectiveness.mean_average_precision << ","
            << r.effectiveness.mean_reciprocal_rank << ","
            << r.effectiveness.ndcg_at_10 << ","
            << r.effectiveness.dcg_at_10 << ","
            << std::setprecision(2) << r.avg_retrieval_time_ms << ","
            << r.avg_reranking_time_ms << ","
            << r.median_retrieval_time_ms << ","
            << r.p95_retrieval_time_ms << "\n";
    }
    
    ofs.close();
    std::cout << "Exported aggregate results to " << filename << std::endl;
}

void BenchmarkSuite::export_query_metrics_csv(
    const BenchmarkResults& results,
    const std::string& filename
) {
    std::ofstream ofs(filename);
    if (!ofs) {
        std::cerr << "Error: Cannot open " << filename << " for writing" << std::endl;
        return;
    }
    
    // CSV header
    ofs << "query_id,num_candidates,retrieval_time_ms,reranking_time_ms,"
        << "precision_at_10,average_precision,reciprocal_rank\n";
    
    // Data rows
    for (const auto& qm : results.query_metrics) {
        ofs << qm.query_id << ","
            << qm.num_candidates << ","
            << qm.retrieval_time_ms << ","
            << qm.reranking_time_ms << ","
            << std::fixed << std::setprecision(4) << qm.precision_at_10 << ","
            << qm.average_precision << ","
            << qm.reciprocal_rank << "\n";
    }
    
    ofs.close();
    std::cout << "Exported per-query metrics to " << filename << std::endl;
}