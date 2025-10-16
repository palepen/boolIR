#include "system_controller.h"
#include "benchmark_suite.h"
#include "data_loader.h"
#include "document_store.h"
#include "reranking/neural_reranker.h"
#include "indexing/bsbi_indexer.h"
#include <iostream>
#include <string>
#include <filesystem>
#include <chrono>
#include <thread>
#include <algorithm>
#include <sstream>
#include <iomanip>

namespace fs = std::filesystem;

void print_separator(char ch = '=', int width = 80) {
    std::cout << std::string(width, ch) << std::endl;
}

static std::string truncate_to_words(const std::string &text, size_t max_words) {
    std::istringstream iss(text);
    std::ostringstream oss;
    std::string word;
    for (size_t count = 0; count < max_words && iss >> word; ++count) {
        oss << (count > 0 ? " " : "") << word;
    }
    return oss.str();
}

int main(int argc, char **argv) {
    print_separator();
    std::cout << "High-Performance IR System (Pure Boolean + Dynamic Retrieval)" << std::endl;
    print_separator();

    const std::string corpus_dir = "data/cord19-trec-covid_corpus_batched";
    const std::string topics_path = "data/topics.cord19-trec-covid.txt";
    const std::string qrels_path = "data/qrels.cord19-trec-covid.txt";
    const std::string synonym_path = "data/synonyms.txt";
    const std::string model_path = "models/bert_model.pt";
    const std::string vocab_path = "models/vocab.txt";
    const std::string index_path = "index";
    const std::string temp_path = "index/temp";

    bool build_index_mode = false;
    bool run_benchmark_mode = false;
    bool benchmark_indexing = false;
    bool run_interactive_mode = false;
    size_t num_shards = 64;

    BenchmarkConfig config;
    config.num_cpu_workers = std::thread::hardware_concurrency();
    config.label = "default_run";
    config.use_partitioned = false; // This flag can be deprecated or repurposed if needed
    config.num_partitions = num_shards; // Use one variable for shard/partition count

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--build-index") build_index_mode = true;
        else if (arg == "--benchmark") run_benchmark_mode = true;
        else if (arg == "--benchmark-indexing") benchmark_indexing = true;
        else if (arg == "--interactive") run_interactive_mode = true;
        else if (arg == "--shards" && i + 1 < argc) {
            num_shards = std::stoi(argv[++i]);
            config.num_partitions = num_shards;
        } else if (arg == "--label" && i + 1 < argc) {
            config.label = argv[++i];
        } else if (arg == "--cpu-workers" && i + 1 < argc) {
            config.num_cpu_workers = std::stoi(argv[++i]);
        }
    }

    if (!build_index_mode && !run_benchmark_mode && !benchmark_indexing && !run_interactive_mode) {
        run_interactive_mode = true; // Default to interactive mode
    }

    if (benchmark_indexing) {
        // This mode remains a useful utility for analyzing indexing performance
        std::cout << "\n[INDEXING BENCHMARK MODE]" << std::endl;
        print_separator('-');
        auto [documents, doc_name_to_id] = load_trec_documents(corpus_dir);
        
        std::vector<size_t> worker_counts = {1, 2, 4, 8};
        std::cout << "\n=== INDEXING SCALABILITY BENCHMARK ===" << std::endl;
        std::cout << "Workers | Time (ms) | Throughput (docs/s) | Speedup | Efficiency" << std::endl;
        std::cout << "--------|-----------|---------------------|---------|------------" << std::endl;
        double baseline_time = 0.0;

        for (size_t workers : worker_counts) {
             BSBIIndexer indexer(documents, "index_test", "index_test/temp", 256, num_shards, workers);
             auto start = std::chrono::high_resolution_clock::now();
             indexer.build_index();
             auto end = std::chrono::high_resolution_clock::now();
             double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
             if (workers == 1) baseline_time = elapsed_ms;
             double throughput = (documents.size() * 1000.0) / elapsed_ms;
             double speedup = baseline_time / elapsed_ms;
             double efficiency = (speedup / workers) * 100.0;
             std::cout << std::setw(7) << workers << " | "
                       << std::setw(9) << std::fixed << std::setprecision(0) << elapsed_ms << " | "
                       << std::setw(19) << std::fixed << std::setprecision(0) << throughput << " | "
                       << std::setw(7) << std::fixed << std::setprecision(2) << speedup << " | "
                       << std::setw(10) << std::fixed << std::setprecision(1) << efficiency << "%" << std::endl;
             fs::remove_all("index_test");
        }
    }

    if (build_index_mode) {
        std::cout << "\n[INDEXING MODE]" << std::endl;
        print_separator('-');
        auto [documents, doc_name_to_id] = load_trec_documents(corpus_dir);
        BSBIIndexer indexer(documents, index_path, temp_path, 256, num_shards, 0);
        indexer.build_index();
        std::cout << "\nSharded indexing complete (" << num_shards << " shards)." << std::endl;
    }

    if (run_benchmark_mode) {
        if (!fs::exists(index_path + "/shard_0/dict.dat")) {
            std::cerr << "FATAL: Sharded index not found. Run with '--build-index' first." << std::endl;
            return 1;
        }
        
        std::cout << "\n[BENCHMARK MODE]" << std::endl;
        print_separator('-');
        DocumentStore doc_store(index_path);
        auto topics = load_trec_topics(topics_path);
        auto [_, doc_name_to_id] = load_trec_documents(corpus_dir);
        Qrels ground_truth = load_trec_qrels(qrels_path, doc_name_to_id);

        BenchmarkSuite suite(doc_store, topics, ground_truth, model_path, vocab_path, index_path, synonym_path);
        suite.run_integrated_benchmark(config);
    }

    if (run_interactive_mode) {
        std::cout << "\n[INTERACTIVE SEARCH MODE]" << std::endl;
        print_separator('-');

        if (!fs::exists(index_path + "/shard_0/dict.dat")) {
             std::cerr << "FATAL: Sharded index not found. Run with '--build-index' first." << std::endl;
             return 1;
        }
        
        HighPerformanceIRSystem system(index_path, synonym_path, num_shards);
        GpuNeuralReranker gpu_reranker(model_path.c_str(), vocab_path.c_str());
        DocumentStore doc_store(index_path);

        std::string query;
        while (true) {
            std::cout << "\nEnter query (or 'exit' to quit): ";
            std::getline(std::cin, query);
            if (query == "exit" || !std::cin.good()) break;
            if (query.empty()) continue;

            auto start_time = std::chrono::high_resolution_clock::now();

            // 1. Get Pure Boolean Results
            auto candidates = system.search_boolean(query);
            
            std::cout << "\n--- Top 5 Pure Boolean Results (Unranked) ---" << std::endl;
            std::cout << "  Found " << candidates.size() << " total documents." << std::endl;
            for (size_t i = 0; i < std::min((size_t)5, candidates.size()); ++i) {
                std::cout << "  " << (i + 1) << ". DocID: " << candidates[i].doc_id << std::endl;
            }

            const size_t MAX_CANDIDATES_FOR_RERANK = 1000;
            std::cout << "\n(Taking top " << MAX_CANDIDATES_FOR_RERANK << " candidates for reranking...)" << std::endl;

            // 2. Rerank only the top N candidates
            std::vector<Document> docs_to_rerank;
            if (!candidates.empty()) {
                size_t rerank_count = std::min(candidates.size(), MAX_CANDIDATES_FOR_RERANK);
                docs_to_rerank.reserve(rerank_count);
                for (size_t i = 0; i < rerank_count; ++i) {
                    const Document* doc_ptr = doc_store.get_document(candidates[i].doc_id);
                    if (doc_ptr) {
                         docs_to_rerank.emplace_back(candidates[i].doc_id, truncate_to_words(doc_ptr->content, 200));
                    }
                }
            }

            auto reranked = gpu_reranker.rerank_with_chunking(query, docs_to_rerank);
            auto end_time = std::chrono::high_resolution_clock::now();
            double elapsed_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();

            std::cout << "\n--- Top 5 Neurally Reranked Results ---" << std::endl;
            for (size_t i = 0; i < std::min((size_t)5, reranked.size()); ++i) {
                std::cout << "  " << (i + 1) << ". DocID: " << reranked[i].id
                          << " (Score: " << std::fixed << std::setprecision(4)
                          << reranked[i].score << ")" << std::endl;
            }
            std::cout << "\nTotal query time: " << std::fixed << std::setprecision(2) << elapsed_ms << " ms" << std::endl;
            print_separator('-');
        }
    }

    print_separator();
    std::cout << "ALL OPERATIONS COMPLETED SUCCESSFULLY!" << std::endl;
    print_separator();

    return 0;
}