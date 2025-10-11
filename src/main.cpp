#include "system_controller.h"
#include "evaluation/evaluator.h"
#include "data_loader.h"
#include "benchmark_suite.h"
#include "reranking/parallel_gpu_reranking.h"
#include "retrieval/optimized_parallel_retrieval.h"
#include <iostream>
#include <iomanip>
#include <thread>
#include <algorithm>
#include <chrono>

void print_separator(char ch = '=', int width = 80) {
    std::cout << std::string(width, ch) << std::endl;
}

void run_quick_demo(
    const DocumentCollection& documents,
    const std::unordered_map<std::string, std::string>& topics,
    const Qrels& ground_truth,
    const char* model_path,
    const char* vocab_path
) {
    std::cout << "\n";
    print_separator('=', 80);
    std::cout << " QUICK DEMO: Boolean vs Neural Reranking " << std::endl;
    print_separator('=', 80);
    
    const size_t num_workers = std::thread::hardware_concurrency() * 2;
    
    // Initialize system
    HighPerformanceIRSystem system(num_workers, model_path, vocab_path);
    
    // Build index
    std::cout << "\nBuilding index..." << std::endl;
    auto start_index = std::chrono::high_resolution_clock::now();
    system.build_index(documents);
    auto end_index = std::chrono::high_resolution_clock::now();
    auto index_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_index - start_index).count();
    std::cout << "Index built in " << index_time << " ms" << std::endl;
    
    // Select 5 sample queries for demo
    std::vector<std::pair<std::string, std::string>> sample_queries;
    int count = 0;
    for (const auto& [qid, qtext] : topics) {
        sample_queries.push_back({qid, qtext});
        if (++count >= 5) break;
    }
    
    std::cout << "\n--- Processing " << sample_queries.size() << " Sample Queries in Parallel ---\n" << std::endl;
    
    BatchedGpuReranker gpu_reranker(model_path, vocab_path);
    std::vector<QueryMetrics> query_metrics;
    auto start = std::chrono::high_resolution_clock::now();
    auto search_results = process_queries_parallel_batched(
        system, gpu_reranker, sample_queries, documents, query_metrics
    );
    auto end = std::chrono::high_resolution_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    for (size_t i = 0; i < sample_queries.size(); ++i) {
        const auto& [qid, qtext] = sample_queries[i];
        const auto& results = search_results[qid];
        const auto& metrics = query_metrics[i];
        
        std::cout << "Query: \"" << qtext << "\"" << std::endl;
        std::cout << "  Boolean: " << metrics.num_candidates << " docs in " 
                  << metrics.retrieval_time_ms << " ms" << std::endl;
        std::cout << "  Reranking: " << results.size() << " docs in " 
                  << metrics.reranking_time_ms << " ms" << std::endl;
        
        if (!results.empty()) {
            std::cout << "  Top 3 results: ";
            for (size_t j = 0; j < std::min(size_t(3), results.size()); ++j) {
                std::cout << results[j].doc_id << "(" 
                          << std::fixed << std::setprecision(3) 
                          << results[j].score << ") ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
    
    std::cout << "Total processing time: " << total_time << " ms" << std::endl;
}

void run_comprehensive_benchmarks(
    const DocumentCollection& documents,
    const std::unordered_map<std::string, std::string>& topics,
    const Qrels& ground_truth,
    const char* model_path,
    const char* vocab_path
) {
    std::cout << "\n";
    print_separator('=', 80);
    std::cout << " COMPREHENSIVE BENCHMARK SUITE " << std::endl;
    print_separator('=', 80);
    
    // Initialize benchmark suite
    BenchmarkSuite suite(documents, topics, ground_truth, model_path, vocab_path);
    
    // Determine worker counts to test
    size_t max_workers = std::thread::hardware_concurrency() * 2;
    std::vector<size_t> worker_counts;
    
    // Test: 1, 2, 4, 8, ..., up to max
    for (size_t w = 1; w <= max_workers; w *= 2) {
        worker_counts.push_back(w);
    }
    // Add max_workers if not already included
    if (worker_counts.back() != max_workers) {
        worker_counts.push_back(max_workers);
    }
    
    std::cout << "\nWorker counts to test: ";
    for (size_t w : worker_counts) {
        std::cout << w << " ";
    }
    std::cout << std::endl;
    
    // ========================================================================
    // TEST 1: Scalability Test - Boolean Retrieval
    // ========================================================================
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "TEST 1: SCALABILITY - BOOLEAN RETRIEVAL" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    
    auto boolean_scalability = suite.run_scalability_test(false, worker_counts);
    suite.export_to_csv(boolean_scalability, "results/boolean_scalability.csv");
    
    // ========================================================================
    // TEST 2: Scalability Test - Neural Reranking
    // ========================================================================
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "TEST 2: SCALABILITY - NEURAL RERANKING" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    
    auto reranking_scalability = suite.run_scalability_test(true, worker_counts);
    suite.export_to_csv(reranking_scalability, "results/reranking_scalability.csv");
    
    // ========================================================================
    // TEST 3: Comparison Test - Boolean vs Reranking at Max Workers
    // ========================================================================
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "TEST 3: COMPARISON - BOOLEAN vs RERANKING (" << max_workers << " workers)" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    
    auto comparison_results = suite.run_comparison_test(max_workers);
    suite.export_to_csv(comparison_results, "results/comparison.csv");
    
    // Combine all results
    std::vector<BenchmarkResults> all_results;
    all_results.insert(all_results.end(), boolean_scalability.begin(), boolean_scalability.end());
    all_results.insert(all_results.end(), reranking_scalability.begin(), reranking_scalability.end());
    
    suite.export_to_csv(all_results, "results/all_benchmarks.csv");
    
    std::cout << "\n All benchmarks completed!" << std::endl;
    std::cout << "\n📊 Results saved to:" << std::endl;
    std::cout << "  • results/boolean_scalability.csv" << std::endl;
    std::cout << "  • results/reranking_scalability.csv" << std::endl;
    std::cout << "  • results/comparison.csv" << std::endl;
    std::cout << "  • results/all_benchmarks.csv" << std::endl;
    std::cout << "  • results/boolean_query_metrics.csv" << std::endl;
    std::cout << "  • results/reranking_query_metrics.csv" << std::endl;
    
    std::cout << "\n📈 Generate visualizations with:" << std::endl;
    std::cout << "  python3 scripts/visualize_benchmarks.py \\" << std::endl;
    std::cout << "    --results results/all_benchmarks.csv \\" << std::endl;
    std::cout << "    --query-metrics results/reranking_query_metrics.csv \\" << std::endl;
    std::cout << "    --output-dir results/plots" << std::endl;
}

int main(int argc, char** argv) {
    // Parse command line arguments
    bool run_full_benchmark = false;
    bool run_demo = true;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--benchmark" || arg == "-b") {
            run_full_benchmark = true;
            run_demo = false;
        } else if (arg == "--demo" || arg == "-d") {
            run_demo = true;
            run_full_benchmark = false;
        } else if (arg == "--all" || arg == "-a") {
            run_demo = true;
            run_full_benchmark = true;
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [OPTIONS]" << std::endl;
            std::cout << "\nOptions:" << std::endl;
            std::cout << "  -d, --demo         Run quick demo (default)" << std::endl;
            std::cout << "  -b, --benchmark    Run comprehensive benchmarks" << std::endl;
            std::cout << "  -a, --all          Run both demo and benchmarks" << std::endl;
            std::cout << "  -h, --help         Show this help message" << std::endl;
            std::cout << "\nExamples:" << std::endl;
            std::cout << "  " << argv[0] << " --demo" << std::endl;
            std::cout << "  " << argv[0] << " --benchmark" << std::endl;
            std::cout << "  " << argv[0] << " --all" << std::endl;
            return 0;
        }
    }
    
    std::cout << "\n";
    print_separator('=', 80);
    std::cout << " High-Performance IR System - TREC COVID Evaluation " << std::endl;
    print_separator('=', 80);
    
    // ============================================================================
    // LOAD DATA
    // ============================================================================
    std::cout << "\n[LOADING DATA]" << std::endl;
    print_separator('-', 80);
    
    const char* model_path = "models/bert_model.onnx";
    const char* vocab_path = "models/vocab.txt";
    const char* corpus_dir = "data/cord19-trec-covid_corpus_batched";
    const char* topics_path = "data/topics.cord19-trec-covid.txt";
    const char* qrels_path = "data/qrels.cord19-trec-covid.txt";
    
    auto start_load = std::chrono::high_resolution_clock::now();
    
    auto [documents, doc_name_to_id] = load_trec_documents(corpus_dir);
    auto topics = load_trec_topics(topics_path);
    Qrels ground_truth = load_trec_qrels(qrels_path, doc_name_to_id);
    
    auto end_load = std::chrono::high_resolution_clock::now();
    auto load_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_load - start_load).count();
    
    std::cout << "Loaded " << documents.size() << " documents" << std::endl;
    std::cout << "Loaded " << topics.size() << " queries" << std::endl;
    std::cout << "Loaded qrels for " << ground_truth.size() << " queries" << std::endl;
    std::cout << "Data loading completed in " << load_time << " ms" << std::endl;
    
    // ============================================================================
    // RUN SELECTED MODE
    // ============================================================================
    
    if (run_demo) {
        run_quick_demo(documents, topics, ground_truth, model_path, vocab_path);
    }
    
    if (run_full_benchmark) {
        run_comprehensive_benchmarks(documents, topics, ground_truth, model_path, vocab_path);
    }
    
    std::cout << "\n";
    print_separator('=', 80);
    std::cout << " ALL OPERATIONS COMPLETED SUCCESSFULLY!" << std::endl;
    print_separator('=', 80);
    std::cout << std::endl;
    
    return 0;
}