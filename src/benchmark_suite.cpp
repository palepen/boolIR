#include "benchmark_suite.h"
#include "system_controller.h"
#include "reranking/neural_reranker.h"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <numeric>
#include <algorithm>
#include <fstream>
#include <cilk/cilk.h>
#include <cilk/cilk_stub.h>
#include <filesystem>
#include <sstream>

// Helper to truncate document content for the reranker
static std::string truncate_to_words(const std::string &text, size_t max_words)
{
    std::istringstream iss(text);
    std::ostringstream oss;
    std::string word;
    for (size_t count = 0; count < max_words && iss >> word; ++count)
    {
        oss << (count > 0 ? " " : "") << word;
    }
    return oss.str();
}

BenchmarkSuite::BenchmarkSuite(
    const DocumentStore &doc_store,
    const std::unordered_map<std::string, std::string> &topics,
    const Qrels &ground_truth,
    const std::string &model_path,
    const std::string &vocab_path,
    const std::string &index_path,
    const std::string &synonym_path) : doc_store_(doc_store),
                                       topics_(topics),
                                       ground_truth_(ground_truth),
                                       model_path_(model_path),
                                       vocab_path_(vocab_path),
                                       index_path_(index_path),
                                       synonym_path_(synonym_path) {}

void BenchmarkSuite::run_integrated_benchmark(const BenchmarkConfig &config)
{
    std::cout << "\n========================================" << std::endl;
    std::cout << "Running Integrated Benchmark: " << config.label << std::endl;
    std::cout << "  -> CPU Workers: " << config.num_cpu_workers << std::endl;
    std::cout << "========================================" << std::endl;

    // Initialize the system with the appropriate number of shards
    size_t num_shards = config.use_partitioned ? config.num_partitions : 64;
    HighPerformanceIRSystem system(index_path_, synonym_path_, num_shards);

    std::vector<std::pair<std::string, std::string>> queries;
    queries.reserve(topics_.size());
    for (const auto &[qid, qtext] : topics_)
        queries.emplace_back(qid, qtext);

    // --- STAGE 1: BOOLEAN RETRIEVAL (TIMED) ---
    std::cout << "\n--- Stage 1: Executing Boolean Retrieval ---" << std::endl;
    auto start_retrieval_phase = std::chrono::high_resolution_clock::now();

    std::vector<std::pair<std::string, std::vector<SearchResult>>> boolean_results_vec(queries.size());
    std::vector<QueryMetrics> query_metrics(queries.size());

    cilk_for(size_t i = 0; i < queries.size(); ++i)
    {
        const auto &[qid, qtext] = queries[i];
        auto start_q = std::chrono::high_resolution_clock::now();
        auto candidates = system.search_boolean(qtext);
        auto end_q = std::chrono::high_resolution_clock::now();

        boolean_results_vec[i] = {qid, candidates};
        query_metrics[i] = {qid, candidates.size(), std::chrono::duration<double, std::milli>(end_q - start_q).count(), 0.0};
    }
    auto end_retrieval_phase = std::chrono::high_resolution_clock::now();

    // --- Process and Log Boolean Results ---
    BenchmarkResults bool_results;
    bool_results.config = config;
    bool_results.query_metrics = query_metrics;
    bool_results.query_processing_time_ms = std::chrono::duration<double, std::milli>(end_retrieval_phase - start_retrieval_phase).count();

    std::unordered_map<std::string, std::vector<SearchResult>> bool_map;
    for (const auto &pair : boolean_results_vec)
        bool_map[pair.first] = pair.second;

    Evaluator evaluator(ground_truth_);
    bool_results.effectiveness = evaluator.evaluate(bool_map);
    bool_results.throughput_queries_per_sec = (topics_.size() * 1000.0) / bool_results.query_processing_time_ms;
    calculate_statistics(bool_results);
    export_to_csv(bool_results, "results/all_benchmarks.csv");
    std::cout << "  -> Boolean Stage Complete. Throughput: " << std::fixed << std::setprecision(2) << bool_results.throughput_queries_per_sec << " q/s" << std::endl;

    // --- STAGE 2: NEURAL RERANKING (TIMED) ---
    std::cout << "\n--- Stage 2: Executing Neural Reranking ---" << std::endl;
    GpuNeuralReranker gpu_reranker(model_path_.c_str(), vocab_path_.c_str());

    auto start_rerank_phase = std::chrono::high_resolution_clock::now();
    std::vector<std::pair<std::string, std::vector<SearchResult>>> reranked_results_vec(queries.size());
    const auto &doc_map = doc_store_.get_all();

    const size_t MAX_CANDIDATES_FOR_RERANK = 2000;
    std::cout << "\n(Taking top " << MAX_CANDIDATES_FOR_RERANK << " candidates for reranking...)" << std::endl;

    
    cilk_for(size_t i = 0; i < queries.size(); ++i)
    {
        const auto &[qid, qtext] = queries[i];
        const auto &candidates = boolean_results_vec[i].second;

        std::vector<Document> candidate_docs;
        size_t rerank_count = std::min(candidates.size(), MAX_CANDIDATES_FOR_RERANK);
        candidate_docs.reserve(rerank_count);
        for (size_t k = 0; k < rerank_count; k++)
        {
            auto it = doc_map.find(candidates[k].doc_id);
            if (it != doc_map.end())
            {
                candidate_docs.emplace_back(candidates[k].doc_id, truncate_to_words(it->second.content, 256));
                
            }
        }

        auto start_rerank_q = std::chrono::high_resolution_clock::now();
        auto reranked_scored = gpu_reranker.rerank_with_chunking(qtext, candidate_docs);
        auto end_rerank_q = std::chrono::high_resolution_clock::now();

        std::vector<SearchResult> final_results;
        final_results.reserve(reranked_scored.size());
        for (const auto &sd : reranked_scored)
            final_results.push_back({sd.id, sd.score});
        reranked_results_vec[i] = {qid, final_results};
        query_metrics[i].reranking_time_ms = std::chrono::duration<double, std::milli>(end_rerank_q - start_rerank_q).count();
    }
    auto end_rerank_phase = std::chrono::high_resolution_clock::now();

    // --- Process and Log Rerank Results ---
    BenchmarkResults rerank_results;
    rerank_results.config = config;
    rerank_results.config.label += "_Rerank"; // Append suffix
    rerank_results.query_metrics = query_metrics;
    // End-to-end time is boolean time + reranking time
    rerank_results.query_processing_time_ms = bool_results.query_processing_time_ms + std::chrono::duration<double, std::milli>(end_rerank_phase - start_rerank_phase).count();

    std::unordered_map<std::string, std::vector<SearchResult>> rerank_map;
    for (const auto &pair : reranked_results_vec)
        rerank_map[pair.first] = pair.second;

    rerank_results.effectiveness = evaluator.evaluate(rerank_map);
    rerank_results.throughput_queries_per_sec = (topics_.size() * 1000.0) / rerank_results.query_processing_time_ms;
    calculate_statistics(rerank_results);
    export_to_csv(rerank_results, "results/all_benchmarks.csv");
    std::cout << "  -> Reranking Stage Complete. End-to-End Throughput: " << std::fixed << std::setprecision(2) << rerank_results.throughput_queries_per_sec << " q/s" << std::endl;

    print_comparison(bool_results, rerank_results);
}

void BenchmarkSuite::calculate_statistics(BenchmarkResults &results)
{
    if (results.query_metrics.empty())
        return;

    double sum_retrieval_ms = 0.0;
    double sum_reranking_ms = 0.0;
    std::vector<double> total_latencies;

    for (const auto &qm : results.query_metrics)
    {
        sum_retrieval_ms += qm.retrieval_time_ms;
        sum_reranking_ms += qm.reranking_time_ms;
        total_latencies.push_back(qm.retrieval_time_ms + qm.reranking_time_ms);
    }

    size_t n = results.query_metrics.size();
    results.avg_retrieval_time_ms = sum_retrieval_ms / n;
    results.avg_reranking_time_ms = sum_reranking_ms / n;

    std::sort(total_latencies.begin(), total_latencies.end());
    if (n > 0)
    {
        results.median_latency_ms = (n % 2 == 1) ? total_latencies[n / 2] : (total_latencies[n / 2 - 1] + total_latencies[n / 2]) / 2.0;
        results.p95_latency_ms = (n > 20) ? total_latencies[static_cast<size_t>(n * 0.95)] : total_latencies.back();
    }
}

void BenchmarkSuite::export_to_csv(const BenchmarkResults &result, const std::string &filename)
{
    bool file_exists = std::filesystem::exists(filename);
    std::ofstream ofs(filename, std::ios_base::app);
    if (!ofs)
    {
        std::cerr << "Error: cannot open " << filename << std::endl;
        return;
    }

    if (!file_exists)
    {
        ofs << "label,num_cpu_workers,use_reranking,query_processing_time_ms,throughput_qps,precision_at_10,map,mrr,ndcg_at_10,avg_retrieval_ms,avg_reranking_ms,median_latency_ms,p95_latency_ms\n";
    }

    const auto &r = result;
    ofs << r.config.label << ","
        << r.config.num_cpu_workers << ","
        << (r.config.label.find("_Rerank") != std::string::npos) << "," // Infer use_reranking
        << r.query_processing_time_ms << ","
        << r.throughput_queries_per_sec << ","
        << r.effectiveness.precision_at_10 << ","
        << r.effectiveness.mean_average_precision << ","
        << r.effectiveness.mean_reciprocal_rank << ","
        << r.effectiveness.ndcg_at_10 << ","
        << r.avg_retrieval_time_ms << ","
        << r.avg_reranking_time_ms << ","
        << r.median_latency_ms << ","
        << r.p95_latency_ms << "\n";
}

void BenchmarkSuite::print_comparison(const BenchmarkResults &bool_res, const BenchmarkResults &rerank_res) const
{
    std::cout << "\n"
              << std::string(80, '=') << std::endl;
    std::cout << "DETAILED COMPARISON: Boolean vs Reranking ("
              << bool_res.config.num_cpu_workers << " CPU workers)" << std::endl;
    std::cout << std::string(80, '=') << std::endl;

    std::cout << "\nTHROUGHPUT:" << std::endl;
    std::cout << "  Boolean (Retrieval Only): " << std::fixed << std::setprecision(2) << bool_res.throughput_queries_per_sec << " q/s" << std::endl;
    std::cout << "  Reranking (End-to-End):   " << std::fixed << std::setprecision(2) << rerank_res.throughput_queries_per_sec << " q/s" << std::endl;

    std::cout << "\nEFFECTIVENESS:" << std::endl;
    std::cout << "  Metric    | Boolean | Reranking | Change" << std::endl;
    std::cout << "  ----------|---------|-----------|--------" << std::endl;

    auto print_metric = [](const std::string &name, double bool_val, double rerank_val)
    {
        double rel_change = (bool_val > 0) ? ((rerank_val - bool_val) / bool_val * 100.0) : 0.0;
        std::cout << "  " << std::left << std::setw(10) << name
                  << "| " << std::fixed << std::setprecision(4) << bool_val
                  << " | " << std::fixed << std::setprecision(4) << rerank_val
                  << " | " << std::showpos << std::fixed << std::setprecision(1) << rel_change << "%" << std::noshowpos << std::endl;
    };

    print_metric("P@10", bool_res.effectiveness.precision_at_10, rerank_res.effectiveness.precision_at_10);
    print_metric("MAP", bool_res.effectiveness.mean_average_precision, rerank_res.effectiveness.mean_average_precision);
    print_metric("NDCG@10", bool_res.effectiveness.ndcg_at_10, rerank_res.effectiveness.ndcg_at_10);

    std::cout << "\nLATENCY (Per Query):" << std::endl;
    std::cout << "  Median (Boolean): " << std::fixed << std::setprecision(2) << bool_res.median_latency_ms << " ms" << std::endl;
    std::cout << "  Median (End-to-End): " << std::fixed << std::setprecision(2) << rerank_res.median_latency_ms << " ms" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
}