#include "system_controller.h"
#include "benchmark_suite.h"
#include "data_loader.h"
#include "document_store.h"
#include "common/utils.h"
#include "config.h"
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

const size_t MAX_CANDIDATES_FOR_RERANK = 1024;
namespace fs = std::filesystem;

void print_separator(char ch = '=', int width = 80)
{
    std::cout << std::string(width, ch) << std::endl;
}

int main(int argc, char **argv)
{
    print_separator();
    std::cout << "High-Performance IR System (Pure Boolean + Dynamic Retrieval)" << std::endl;
    print_separator();

    bool build_index_mode = false;
    bool run_benchmark_mode = false;
    bool benchmark_indexing = false;
    bool run_interactive_mode = false;
    size_t num_shards = Config::DEFAULT_NUM_SHARDS;

    BenchmarkConfig config;
    config.num_cpu_workers = std::thread::hardware_concurrency();
    config.label = "default_run";
    config.use_partitioned = false;     // This flag can be deprecated or repurposed if needed
    config.num_partitions = num_shards; // Use one variable for shard/partition count

    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "--build-index")
            build_index_mode = true;
        else if (arg == "--benchmark")
            run_benchmark_mode = true;
        else if (arg == "--benchmark-indexing")
            benchmark_indexing = true;
        else if (arg == "--interactive")
            run_interactive_mode = true;
        else if (arg == "--shards" && i + 1 < argc)
        {
            num_shards = std::stoi(argv[++i]);
            config.num_partitions = num_shards;
        }
        else if (arg == "--label" && i + 1 < argc)
        {
            config.label = argv[++i];
        }
        else if (arg == "--cpu-workers" && i + 1 < argc)
        {
            config.num_cpu_workers = std::stoi(argv[++i]);
        }
    }

    if (!build_index_mode && !run_benchmark_mode && !benchmark_indexing && !run_interactive_mode)
    {
        run_interactive_mode = true;
    }

    if (benchmark_indexing)
    {
        std::cout << "\n[INDEXING BENCHMARK MODE]" << std::endl;
        print_separator('-');

        auto doc_load_result = load_trec_documents(Config::CORPUS_DIR);
        std::vector<size_t> worker_counts = {1, 2, 4, 8};
        std::cout << "\n=== INDEXING SCALABILITY BENCHMARK ===" << std::endl;
        std::cout << "Workers | Time (ms) | Throughput (docs/s) | Speedup | Efficiency" << std::endl;
        std::cout << "--------|-----------|---------------------|---------|------------" << std::endl;
        double baseline_time = 0.0;

        for (size_t workers : worker_counts)
        {
            BSBIIndexer indexer(doc_load_result.documents, doc_load_result.id_to_doc_name, Config::INDEX_PATH, Config::TEMP_PATH, Config::DEFAULT_BLOCK_SIZE_MB, num_shards, workers);
            auto start = std::chrono::high_resolution_clock::now();
            indexer.build_index();
            auto end = std::chrono::high_resolution_clock::now();
            double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
            if (workers == 1)
                baseline_time = elapsed_ms;
            double throughput = (doc_load_result.documents.size() * 1000.0) / elapsed_ms;
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

    if (build_index_mode)
    {
        std::cout << "\n[INDEXING MODE]" << std::endl;
        print_separator('-');
        // 7. USE CONFIG FOR PATHS AND PARAMS
        auto doc_load_result = load_trec_documents(Config::CORPUS_DIR);
        BSBIIndexer indexer(doc_load_result.documents, doc_load_result.id_to_doc_name, Config::INDEX_PATH, Config::TEMP_PATH, Config::DEFAULT_BLOCK_SIZE_MB, num_shards, 6);
        indexer.build_index();
        std::cout << "\nSharded indexing complete (" << num_shards << " shards)." << std::endl;
    }

    if (run_benchmark_mode)
    {
        // 8. USE CONFIG FOR PATHS
        if (!fs::exists(Config::INDEX_PATH + "/shard_0/dict.dat"))
        {
            std::cerr << "FATAL: Sharded index not found. Run with '--build-index' first." << std::endl;
            return 1;
        }

        std::cout << "\n[BENCHMARK MODE]" << std::endl;
        print_separator('-');

        DocumentStore doc_store(Config::INDEX_PATH);
        auto topics = load_trec_topics(Config::TOPICS_PATH);
        
        Qrels ground_truth = load_trec_qrels(Config::QRELS_PATH, doc_store.get_doc_name_to_id_map());

        BenchmarkSuite suite(doc_store, topics, ground_truth, Config::MODEL_PATH, Config::VOCAB_PATH, Config::INDEX_PATH, Config::SYNONYM_PATH);
        suite.run_integrated_benchmark(config);
    }

    if (run_interactive_mode)
    {
        std::cout << "\n[INTERACTIVE SEARCH MODE]" << std::endl;
        print_separator('-');

        if (!fs::exists(Config::INDEX_PATH + "/shard_0/dict.dat"))
        {
            std::cerr << "FATAL: Sharded index not found. Run with '--build-index' first." << std::endl;
            return 1;
        }

        HighPerformanceIRSystem system(Config::INDEX_PATH, Config::SYNONYM_PATH, num_shards);
        GpuNeuralReranker gpu_reranker(Config::MODEL_PATH.c_str(), Config::VOCAB_PATH.c_str(), Config::BATCH_SIZE);
        DocumentStore doc_store(Config::INDEX_PATH);

        std::string query;
        while (true)
        {
            std::cout << "\nEnter query (or 'exit' to quit): ";
            std::getline(std::cin, query);
            if (query == "exit" || !std::cin.good())
                break;
            if (query.empty())
                continue;

            auto start_time = std::chrono::high_resolution_clock::now();

            // 1. Get Pure Boolean Results
            auto candidates = system.search_boolean(query);

            std::cout << "\n--- Top 5 Pure Boolean Results (Unranked) ---" << std::endl;
            std::cout << "  Found " << candidates.size() << " total documents." << std::endl;

            // CHANGED: Display document names
            for (size_t i = 0; i < std::min((size_t)5, candidates.size()); ++i)
            {
                const std::string *doc_name = doc_store.get_document_name(candidates[i].doc_id);
                if (doc_name)
                {
                    std::cout << "  " << (i + 1) << ". Document: " << "./"  << Config::CORPUS_DIR << "/" << *doc_name << ".txt"
                              << " (ID: " << candidates[i].doc_id << ")" << std::endl;
                }
                else
                {
                    std::cout << "  " << (i + 1) << ". DocID: " << candidates[i].doc_id
                              << " (name unavailable)" << std::endl;
                }
            }

            std::cout << "\n(Taking top " << Config::MAX_RERANK_CANDIDATES << " candidates for reranking...)" << std::endl; // <-- USE CONFIG

            // 2. Rerank only the top N candidates
            std::vector<Document> docs_to_rerank;
            if (!candidates.empty())
            {
                // 10. USE CONFIG FOR PARAMS
                size_t rerank_count = std::min(candidates.size(), (size_t)Config::MAX_RERANK_CANDIDATES);
                docs_to_rerank.reserve(rerank_count);
                for (size_t i = 0; i < rerank_count; ++i)
                {
                    const Document *doc_ptr = doc_store.get_document(candidates[i].doc_id);
                    if (doc_ptr)
                    {
                        docs_to_rerank.emplace_back(candidates[i].doc_id, truncate_to_words(doc_ptr->content, Config::DOCUMENT_TRUNCATE_WORDS)); // <-- USE CONFIG
                    }
                }
            }

            auto reranked = gpu_reranker.rerank_with_chunking(query, docs_to_rerank);
            auto end_time = std::chrono::high_resolution_clock::now();
            double elapsed_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();

            std::cout << "\n--- Top 5 Neurally Reranked Results ---" << std::endl;

            // CHANGED: Display document names with scores
            for (size_t i = 0; i < std::min((size_t)5, reranked.size()); ++i)
            {
                const std::string *doc_name = doc_store.get_document_name(reranked[i].id);
                if (doc_name)
                {
                    std::cout << "  " << (i + 1) << ". Document: " << "./" << Config::CORPUS_DIR << "/" << *doc_name << ".txt"
                              << " (ID: " << reranked[i].id
                              << ", Score: " << std::fixed << std::setprecision(4)
                              << reranked[i].score << ")" << std::endl;
                }
                else
                {
                    std::cout << "  " << (i + 1) << ". DocID: " << reranked[i].id
                              << " (Score: " << std::fixed << std::setprecision(4)
                              << reranked[i].score << ", name unavailable)" << std::endl;
                }
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