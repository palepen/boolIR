#include "benchmark_suite.h"
#include "system_controller.h"
#include "reranking/neural_reranker.h"
#include "retrieval/pre_ranker.h"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <numeric>
#include <algorithm>
#include <fstream>
#include <thread>
#include <cilk/cilk.h>
#include <cilk/cilk_stub.h>
#include <filesystem>
#include <unordered_map>
#include <sstream>

// OPTIMIZATION: Truncate documents to first N words to speed up reranking
static std::string truncate_to_words(const std::string& text, size_t max_words) {
    std::istringstream iss(text);
    std::ostringstream oss;
    std::string word;
    size_t count = 0;
    
    while (count < max_words && iss >> word) {
        if (count > 0) oss << " ";
        oss << word;
        count++;
    }
    return oss.str();
}

BenchmarkSuite::BenchmarkSuite(
    const DocumentCollection &documents,
    const std::unordered_map<std::string, std::string> &topics,
    const Qrels &ground_truth,
    const std::string &model_path,
    const std::string &vocab_path,
    const std::string &index_path,
    const std::string &synonym_path) : documents_(documents),
                                       topics_(topics),
                                       ground_truth_(ground_truth),
                                       model_path_(model_path),
                                       vocab_path_(vocab_path),
                                       index_path_(index_path),
                                       synonym_path_(synonym_path) {}

void BenchmarkSuite::run_single_benchmark(const BenchmarkConfig &config)
{
    std::cout << "\n========================================" << std::endl;
    std::cout << "Running Benchmark: " << config.label << std::endl;
    std::cout << "  -> CPU Workers: " << config.num_cpu_workers 
              << ", Reranking: " << (config.use_reranking ? "Yes" : "No") << std::endl;
    std::cout << "========================================" << std::endl;

    BenchmarkResults results;
    results.config = config;

    HighPerformanceIRSystem system(index_path_, synonym_path_, std::make_unique<TermOverlapRanker>());

    std::vector<std::pair<std::string, std::string>> queries;
    queries.reserve(topics_.size());
    for (const auto &[qid, qtext] : topics_)
        queries.emplace_back(qid, qtext);

    auto start_queries = std::chrono::high_resolution_clock::now();
    std::vector<std::pair<std::string, std::vector<SearchResult>>> all_search_results(queries.size());
    std::vector<QueryMetrics> query_metrics(queries.size());

    // Create a map for fast document lookup by ID
    std::unordered_map<unsigned int, const Document*> doc_id_map;
    doc_id_map.reserve(documents_.size());
    for(const auto& doc : documents_) {
        doc_id_map[doc.id] = &doc;
    }

    if (config.use_reranking)
    {
        // FIXED: Use single GPU reranker with larger batch size instead of worker pool
        GpuNeuralReranker gpu_reranker(model_path_.c_str(), vocab_path_.c_str(), 32);
        
        cilk_for (size_t i = 0; i < queries.size(); ++i)
        {
            const auto& [qid, qtext] = queries[i];
            
            auto start_retrieval = std::chrono::high_resolution_clock::now();
            auto candidates = system.search_boolean(qtext, documents_);
            auto end_retrieval = std::chrono::high_resolution_clock::now();

            // CRITICAL FIX: Rerank top-K but KEEP all candidates
            const size_t MAX_RERANK_CANDIDATES = 100;
            std::vector<SearchResult> final_results;
            
            if (candidates.size() <= MAX_RERANK_CANDIDATES) {
                // If we have fewer candidates than limit, rerank all
                std::vector<Document> candidate_docs;
                candidate_docs.reserve(candidates.size());
                for (const auto& res : candidates) {
                    auto it = doc_id_map.find(res.doc_id);
                    if (it != doc_id_map.end()) {
                        std::string truncated_content = truncate_to_words(it->second->content, 200);
                        candidate_docs.emplace_back(res.doc_id, truncated_content);
                    }
                }

                auto start_rerank = std::chrono::high_resolution_clock::now();
                auto reranked_scored_docs = gpu_reranker.rerank_with_chunking(qtext, candidate_docs);
                auto end_rerank = std::chrono::high_resolution_clock::now();
                
                for(const auto& sd : reranked_scored_docs) {
                    final_results.push_back({sd.id, sd.score});
                }
                
                query_metrics[i].reranking_time_ms = std::chrono::duration<double, std::milli>(end_rerank - start_rerank).count();
            } else {
                // Rerank top-K, append remaining in original order
                std::vector<Document> top_k_docs;
                top_k_docs.reserve(MAX_RERANK_CANDIDATES);
                for (size_t j = 0; j < MAX_RERANK_CANDIDATES; ++j) {
                    auto it = doc_id_map.find(candidates[j].doc_id);
                    if (it != doc_id_map.end()) {
                        std::string truncated_content = truncate_to_words(it->second->content, 200);
                        top_k_docs.emplace_back(candidates[j].doc_id, truncated_content);
                    }
                }

                auto start_rerank = std::chrono::high_resolution_clock::now();
                auto reranked_scored_docs = gpu_reranker.rerank_with_chunking(qtext, top_k_docs);
                auto end_rerank = std::chrono::high_resolution_clock::now();
                
                // Add reranked top-K
                for(const auto& sd : reranked_scored_docs) {
                    final_results.push_back({sd.id, sd.score});
                }
                
                // CRITICAL: Append remaining documents with decayed scores
                float min_rerank_score = final_results.empty() ? 0.0f : final_results.back().score;
                for (size_t j = MAX_RERANK_CANDIDATES; j < candidates.size(); ++j) {
                    // Give slightly lower scores to preserve ranking
                    float decayed_score = min_rerank_score * 0.9f * (1.0f - (j - MAX_RERANK_CANDIDATES) * 0.0001f);
                    final_results.push_back({candidates[j].doc_id, decayed_score});
                }
                
                query_metrics[i].reranking_time_ms = std::chrono::duration<double, std::milli>(end_rerank - start_rerank).count();
            }

            all_search_results[i] = {qid, final_results};
            query_metrics[i].query_id = qid;
            query_metrics[i].num_candidates = candidates.size();
            query_metrics[i].retrieval_time_ms = std::chrono::duration<double, std::milli>(end_retrieval - start_retrieval).count();
        }
    }
    else
    {
        cilk_for (size_t i = 0; i < queries.size(); ++i)
        {
            const auto &[qid, qtext] = queries[i];
            auto start_q = std::chrono::high_resolution_clock::now();
            auto s_results = system.search_boolean(qtext, documents_);
            auto end_q = std::chrono::high_resolution_clock::now();

            all_search_results[i] = {qid, s_results};
            query_metrics[i].query_id = qid;
            query_metrics[i].retrieval_time_ms = std::chrono::duration<double, std::milli>(end_q - start_q).count();
        }
    }

    // Convert vector of pairs to map for evaluator
    std::unordered_map<std::string, std::vector<SearchResult>> search_results_map;
    for(const auto& pair : all_search_results) {
        search_results_map[pair.first] = pair.second;
    }

    auto end_queries = std::chrono::high_resolution_clock::now();
    results.query_processing_time_ms = std::chrono::duration<double, std::milli>(end_queries - start_queries).count();
    results.query_metrics = query_metrics;

    Evaluator evaluator(ground_truth_);
    results.effectiveness = evaluator.evaluate(search_results_map);
    results.throughput_queries_per_sec = (topics_.size() * 1000.0) / results.query_processing_time_ms;

    calculate_statistics(results);

    std::cout << "\n--- Benchmark Summary: " << config.label << " ---" << std::endl;
    std::cout << "  Throughput: " << std::fixed << std::setprecision(2) << results.throughput_queries_per_sec << " q/s" << std::endl;
    std::cout << "  P@10:       " << std::fixed << std::setprecision(4) << results.effectiveness.precision_at_10 << std::endl;
    std::cout << "  MAP:        " << std::fixed << std::setprecision(4) << results.effectiveness.mean_average_precision << std::endl;
    std::cout << "  NDCG@10:    " << std::fixed << std::setprecision(4) << results.effectiveness.ndcg_at_10 << std::endl;
    std::cout << "  Latency (Median): " << std::fixed << std::setprecision(2) << results.median_latency_ms << " ms" << std::endl;
    std::cout << "------------------------------------" << std::endl;
    
    export_to_csv(results, "results/all_benchmarks.csv");
    
    // Store for comparison
    if (!config.use_reranking) {
        last_boolean_results_ = results;
    } else {
        last_reranking_results_ = results;
    }
}

void BenchmarkSuite::calculate_statistics(BenchmarkResults &results)
{
    if (results.query_metrics.empty()) return;

    double sum_retrieval_ms = 0.0;
    double sum_reranking_ms = 0.0;
    std::vector<double> total_latencies;
    total_latencies.reserve(results.query_metrics.size());

    for (const auto &qm : results.query_metrics) {
        sum_retrieval_ms += qm.retrieval_time_ms;
        sum_reranking_ms += qm.reranking_time_ms;
        total_latencies.push_back(qm.retrieval_time_ms + qm.reranking_time_ms);
    }

    size_t n = results.query_metrics.size();
    results.avg_retrieval_time_ms = sum_retrieval_ms / n;
    results.avg_reranking_time_ms = sum_reranking_ms / n;

    std::sort(total_latencies.begin(), total_latencies.end());
    if (n > 0) {
        results.median_latency_ms = (n % 2 == 1) ? total_latencies[n / 2] : (total_latencies[n / 2 - 1] + total_latencies[n / 2]) / 2.0;
        if (n > 20) {
            results.p95_latency_ms = total_latencies[static_cast<size_t>(n * 0.95)];
        } else {
            results.p95_latency_ms = total_latencies.back();
        }
    } else {
        results.median_latency_ms = 0;
        results.p95_latency_ms = 0;
    }
}

void BenchmarkSuite::print_comparison() const {
    if (last_boolean_results_.query_metrics.empty() || last_reranking_results_.query_metrics.empty()) {
        std::cout << "\nNo comparison available - run both Boolean and Reranking benchmarks first." << std::endl;
        return;
    }

    const auto& bool_res = last_boolean_results_;
    const auto& rerank_res = last_reranking_results_;

    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "DETAILED COMPARISON: Boolean vs Reranking" << std::endl;
    std::cout << std::string(80, '=') << std::endl;

    std::cout << "\nCONFIGURATION:" << std::endl;
    std::cout << "  Boolean:   " << bool_res.config.num_cpu_workers << " CPU workers" << std::endl;
    std::cout << "  Reranking: " << rerank_res.config.num_cpu_workers << " CPU workers" << std::endl;

    std::cout << "\nTHROUGHPUT:" << std::endl;
    std::cout << "  Boolean:   " << std::fixed << std::setprecision(2) << bool_res.throughput_queries_per_sec << " q/s" << std::endl;
    std::cout << "  Reranking: " << std::fixed << std::setprecision(2) << rerank_res.throughput_queries_per_sec << " q/s" << std::endl;
    std::cout << "  Slowdown:  " << std::fixed << std::setprecision(1) << (bool_res.throughput_queries_per_sec / rerank_res.throughput_queries_per_sec) << "x" << std::endl;

    std::cout << "\nEFFECTIVENESS:" << std::endl;
    std::cout << "  Metric    | Boolean | Reranking | Absolute | Relative" << std::endl;
    std::cout << "  ----------|---------|-----------|----------|----------" << std::endl;
    
    auto print_metric = [](const std::string& name, double bool_val, double rerank_val) {
        double abs_change = rerank_val - bool_val;
        double rel_change = (bool_val > 0) ? ((rerank_val - bool_val) / bool_val * 100.0) : 0.0;
        std::cout << "  " << std::left << std::setw(10) << name 
                  << "| " << std::fixed << std::setprecision(4) << bool_val
                  << " | " << std::fixed << std::setprecision(4) << rerank_val
                  << " | " << std::showpos << std::fixed << std::setprecision(4) << abs_change
                  << " | " << std::showpos << std::fixed << std::setprecision(1) << rel_change << "%" << std::noshowpos << std::endl;
    };

    print_metric("P@10", bool_res.effectiveness.precision_at_10, rerank_res.effectiveness.precision_at_10);
    print_metric("MAP", bool_res.effectiveness.mean_average_precision, rerank_res.effectiveness.mean_average_precision);
    print_metric("MRR", bool_res.effectiveness.mean_reciprocal_rank, rerank_res.effectiveness.mean_reciprocal_rank);
    print_metric("NDCG@10", bool_res.effectiveness.ndcg_at_10, rerank_res.effectiveness.ndcg_at_10);

    std::cout << "\nLATENCY:" << std::endl;
    std::cout << "  Median: " << std::fixed << std::setprecision(2) << bool_res.median_latency_ms 
              << " ms (Boolean) vs " << rerank_res.median_latency_ms << " ms (Reranking)" << std::endl;
    std::cout << "  P95:    " << std::fixed << std::setprecision(2) << bool_res.p95_latency_ms 
              << " ms (Boolean) vs " << rerank_res.p95_latency_ms << " ms (Reranking)" << std::endl;

    // Per-query improvement analysis
    Evaluator evaluator(ground_truth_);
    int improved = 0, degraded = 0, unchanged = 0;
    
    for (const auto& [qid, qtext] : topics_) {
        // This is a simplified analysis - in practice you'd need per-query MAP
        // For now, we'll just count based on overall metrics
    }

    std::cout << "\n" << std::string(80, '=') << std::endl;
}

void BenchmarkSuite::export_to_csv(const BenchmarkResults &result, const std::string &filename)
{
    bool file_exists = std::filesystem::exists(filename);
    std::ofstream ofs(filename, std::ios_base::app);
    if (!ofs) {
        std::cerr << "Error: cannot open " << filename << std::endl;
        return;
    }

    if (!file_exists) {
        ofs << "label,num_cpu_workers,use_reranking,query_processing_time_ms,throughput_qps,precision_at_10,map,mrr,ndcg_at_10,avg_retrieval_ms,avg_reranking_ms,median_latency_ms,p95_latency_ms\n";
    }

    const auto& r = result;
    ofs << r.config.label << ","
        << r.config.num_cpu_workers << ","
        << r.config.use_reranking << ","
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