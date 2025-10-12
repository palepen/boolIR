#include "benchmark_suite.h"
#include "system_controller.h"
#include "reranking/neural_reranker.h"
#include "reranking/parallel_gpu_reranking.h"
#include "reranking/gpu_worker_pool.h"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <numeric>
#include <algorithm>
#include <fstream>
#include <thread>

GpuRerankService::GpuRerankService(const std::string &model_path, const std::string &vocab_path)
    : reranker_(model_path.c_str(), vocab_path.c_str())
{
    std::cout << "Initializing Single-Threaded GPU Rerank Service..." << std::endl;
    worker_thread_ = std::thread(&GpuRerankService::worker_loop, this);
}

GpuRerankService::~GpuRerankService()
{
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        stop_ = true;
    }
    condition_.notify_one();
    if (worker_thread_.joinable())
    {
        worker_thread_.join();
    }
}

std::future<std::vector<ScoredDocument>> GpuRerankService::submit_job(const std::string &query_text, const std::vector<Document> &candidates)
{
    RerankJob job;
    job.query_text = query_text;
    job.candidates = candidates;
    auto future = job.promise.get_future();
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        job_queue_.push(std::move(job));
    }
    condition_.notify_one();
    return future;
}

void GpuRerankService::worker_loop()
{
    while (true)
    {
        RerankJob job;
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            condition_.wait(lock, [this]
                            { return stop_ || !job_queue_.empty(); });
            if (stop_ && job_queue_.empty())
                return;
            job = std::move(job_queue_.front());
            job_queue_.pop();
        }
        try
        {
            auto results = reranker_.rerank(job.query_text, job.candidates);
            job.promise.set_value(std::move(results));
        }
        catch (...)
        {
            job.promise.set_exception(std::current_exception());
        }
    }
}

// Constructor
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

// Main benchmark runner
void BenchmarkSuite::run_full_benchmark()
{
    auto print_separator = []()
    { std::cout << std::string(80, '=') << std::endl; };
    print_separator();
    std::cout << " COMPREHENSIVE BENCHMARK SUITE " << std::endl;
    print_separator();

    size_t max_workers = std::thread::hardware_concurrency();
    if (max_workers > 24)
        max_workers = 24;
    std::vector<size_t> worker_counts;
    for (size_t w = 1; w <= max_workers; w *= 2)
        worker_counts.push_back(w);
    if (worker_counts.empty() || worker_counts.back() != max_workers)
        worker_counts.push_back(max_workers);

    auto boolean_scalability = run_scalability_test(false, worker_counts);
    export_to_csv(boolean_scalability, "results/boolean_scalability.csv");

    auto reranking_scalability = run_scalability_test(true, worker_counts);
    export_to_csv(reranking_scalability, "results/reranking_scalability.csv");

    auto comparison_results = run_comparison_test(max_workers);
    export_to_csv(comparison_results, "results/comparison.csv");

    std::vector<BenchmarkResults> all_results;
    all_results.insert(all_results.end(), boolean_scalability.begin(), boolean_scalability.end());
    all_results.insert(all_results.end(), reranking_scalability.begin(), reranking_scalability.end());
    export_to_csv(all_results, "results/all_benchmarks.csv");

    std::cout << "\n All benchmarks completed!" << std::endl;
}

// Core logic for a single benchmark configuration
BenchmarkResults BenchmarkSuite::run_benchmark(const BenchmarkConfig &config)
{
    std::cout << "\n========================================" << std::endl;
    std::cout << "Running Benchmark: " << config.label << std::endl;
    std::cout << "========================================" << std::endl;

    BenchmarkResults results;
    results.config = config;

    HighPerformanceIRSystem system(index_path_, synonym_path_);

    std::vector<std::pair<std::string, std::string>> queries;
    queries.reserve(topics_.size());
    for (const auto &[qid, qtext] : topics_)
        queries.emplace_back(qid, qtext);

    auto start_queries = std::chrono::high_resolution_clock::now();
    std::unordered_map<std::string, std::vector<SearchResult>> search_results;

    if (config.use_reranking)
    {
        GpuRerankService rerank_service(model_path_, vocab_path_);
        search_results = process_queries_parallel_batched(system, rerank_service, queries, documents_, results.query_metrics);
    }
    else
    {
        // **THE FIX IS HERE**: Changed 'query_metrics' to 'results.query_metrics'
        results.query_metrics.reserve(topics_.size());
        for (const auto &[qid, qtext] : queries)
        {
            QueryMetrics qm;
            auto start_q = std::chrono::high_resolution_clock::now();
            auto s_results = system.search(qtext, documents_);
            auto end_q = std::chrono::high_resolution_clock::now();

            qm.query_id = qid;
            qm.retrieval_time_ms = std::chrono::duration<double, std::milli>(end_q - start_q).count();
            search_results[qid] = s_results;
            results.query_metrics.push_back(qm);
        }
    }

    auto end_queries = std::chrono::high_resolution_clock::now();
    results.query_processing_time_ms = std::chrono::duration<double, std::milli>(end_queries - start_queries).count();

    Evaluator evaluator(ground_truth_);
    results.effectiveness = evaluator.evaluate(search_results);
    results.throughput_queries_per_sec = (topics_.size() * 1000.0) / results.query_processing_time_ms;

    calculate_statistics(results);

    // Print summary
    std::cout << "\n--- Benchmark Summary: " << config.label << " ---" << std::endl;
    std::cout << "  Throughput: " << std::fixed << std::setprecision(2) << results.throughput_queries_per_sec << " q/s" << std::endl;
    std::cout << "  P@10:       " << std::fixed << std::setprecision(4) << results.effectiveness.precision_at_10 << std::endl;
    std::cout << "  MAP:        " << results.effectiveness.mean_average_precision << std::endl;
    std::cout << "------------------------------------" << std::endl;

    return results;
}

std::vector<BenchmarkResults> BenchmarkSuite::run_scalability_test(bool use_reranking, const std::vector<size_t> &worker_counts)
{
    std::vector<BenchmarkResults> r;
    for (size_t n : worker_counts)
    {
        BenchmarkConfig c;
        c.num_workers = n;
        c.use_reranking = use_reranking;
        c.label = (use_reranking ? "Reranking_" : "Boolean_") + std::to_string(n) + "w";
        r.push_back(run_benchmark(c));
    }
    return r;
}

std::vector<BenchmarkResults> BenchmarkSuite::run_comparison_test(size_t num_workers)
{
    std::vector<BenchmarkResults> r;
    BenchmarkConfig c_b;
    c_b.num_workers = num_workers;
    c_b.use_reranking = false;
    c_b.label = "Boolean_" + std::to_string(num_workers) + "w";
    auto res_b = run_benchmark(c_b);

    BenchmarkConfig c_r;
    c_r.num_workers = num_workers;
    c_r.use_reranking = true;
    c_r.label = "Reranking_" + std::to_string(num_workers) + "w";
    auto res_r = run_benchmark(c_r);

    r.push_back(res_b);
    r.push_back(res_r);

    std::cout << "\n--- Comparison Summary (" << num_workers << " workers) ---" << std::endl;
    auto pr = [](const std::string &m, double v1, double v2)
    { double impr = ((v2 - v1) / (v1 + 1e-9)) * 100.0; std::cout << "  " << std::setw(15) << std::left << m << ": " << std::setw(10) << std::fixed << std::setprecision(4) << v1 << " -> " << std::setw(10) << v2 << " (" << std::showpos << std::setprecision(1) << impr << "%)" << std::noshowpos << std::endl; };
    pr("Throughput", res_b.throughput_queries_per_sec, res_r.throughput_queries_per_sec);
    pr("P@10", res_b.effectiveness.precision_at_10, res_r.effectiveness.precision_at_10);
    pr("MAP", res_b.effectiveness.mean_average_precision, res_r.effectiveness.mean_average_precision);
    pr("MRR", res_b.effectiveness.mean_reciprocal_rank, res_r.effectiveness.mean_reciprocal_rank);
    std::cout << "------------------------------------------" << std::endl;
    return r;
}

void BenchmarkSuite::calculate_statistics(BenchmarkResults &results)
{
    if (results.query_metrics.empty())
        return;
    double sum_r = 0.0, sum_rr = 0.0;
    std::vector<double> total_t;
    for (const auto &qm : results.query_metrics)
    {
        sum_r += qm.retrieval_time_ms;
        sum_rr += qm.reranking_time_ms;
        total_t.push_back(qm.retrieval_time_ms + qm.reranking_time_ms);
    }
    size_t n = results.query_metrics.size();
    results.avg_retrieval_time_ms = sum_r / n;
    results.avg_reranking_time_ms = sum_rr / n;
    std::sort(total_t.begin(), total_t.end());
    results.median_retrieval_time_ms = n > 0 ? total_t[n / 2] : 0;
    results.p95_retrieval_time_ms = n > 0 ? total_t[static_cast<size_t>(n * 0.95)] : 0;
}

void BenchmarkSuite::export_to_csv(const std::vector<BenchmarkResults> &results, const std::string &filename)
{
    std::ofstream ofs(filename);
    if (!ofs)
    {
        std::cerr << "Error: cannot open " << filename << std::endl;
        return;
    }
    ofs << "label,num_workers,use_reranking,query_processing_time_ms,throughput_qps,precision_at_10,map,mrr\n";
    for (const auto &r : results)
    {
        ofs << r.config.label << ","
            << r.config.num_workers << ","
            << r.config.use_reranking << ","
            << r.query_processing_time_ms << ","
            << r.throughput_queries_per_sec << ","
            << r.effectiveness.precision_at_10 << ","
            << r.effectiveness.mean_average_precision << ","
            << r.effectiveness.mean_reciprocal_rank << "\n";
    }
    std::cout << "Exported results to " << filename << std::endl;
}