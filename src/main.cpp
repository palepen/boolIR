#include "system_controller.h"
#include "benchmark_suite.h"
#include "data_loader.h"
#include "document_store.h"
#include "common/utils.h"
#include "config.h"
#include "reranking/neural_reranker.h"
#include "indexing/indexer.h"  
#include "indexing/document_stream.h"          
#include <iostream>
#include <string>
#include <filesystem>
#include <chrono>
#include <cilk/cilk_api.h>
#include <thread>
#include <algorithm>
#include <sstream>
#include <iomanip>


namespace fs = std::filesystem;

void print_separator(char ch = '=', int width = 80)
{
    std::cout << std::string(width, ch) << std::endl;
}

int main(int argc, char **argv)
{
    print_separator();
    std::cout << "High-Performance IR System (Streaming Architecture)" << std::endl;
    std::cout << "Memory-Efficient: Can index corpora larger than RAM" << std::endl;
    print_separator();

    bool build_index_mode = false;
    bool run_benchmark_mode = false;
    bool benchmark_indexing = false;
    bool run_interactive_mode = false;
    size_t num_shards = Config::DEFAULT_NUM_SHARDS;

    BenchmarkConfig config;
    config.num_cpu_workers = __cilkrts_get_nworkers();
    config.label = "default_run";
    config.use_partitioned = false;
    config.num_partitions = num_shards;

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
        else if (arg == "--log-query")
        {
            config.print_log = true;
        }
    }

    if (!build_index_mode && !run_benchmark_mode && !benchmark_indexing && !run_interactive_mode)
    {
        run_interactive_mode = true;
    }

    if (benchmark_indexing)
    {
        std::cout << "\n[INDEXING BENCHMARK MODE - STREAMING]" << std::endl;
        print_separator('-');

        std::string csv_path = Config::INDEXING_CSV_PATH;
        bool file_exists = fs::exists(csv_path);

        std::ofstream index_ofs(csv_path, std::ios_base::app);
        if (!index_ofs)
        {
            std::cerr << "Error: Cannot open " << csv_path << " for writing." << std::endl;
        }
        else if (!file_exists)
        {
            index_ofs << "num_cpu_workers,indexing_time_ms,throughput_docs_per_sec\n";
        }

        std::cout << "\n=== INDEXING SCALABILITY BENCHMARK (STREAMING) ===" << std::endl;
        std::cout << "Running benchmark for " << config.num_cpu_workers << " workers..." << std::endl;
        std::cout << "(Using memory-efficient streaming approach)" << std::endl;
        
        // NEW: Use DocumentStream instead of loading all into RAM
        DocumentStream doc_stream(Config::CORPUS_DIR);

        Indexer indexer(
            doc_stream,
            Config::INDEX_PATH, 
            Config::TEMP_PATH, 
            Config::DEFAULT_BLOCK_SIZE_MB, 
            num_shards, 
            config.num_cpu_workers
        );
        
        auto start = std::chrono::high_resolution_clock::now();
        indexer.build_index();
        auto end = std::chrono::high_resolution_clock::now();

        double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
        double throughput = (doc_stream.size() * 1000.0) / elapsed_ms;

        std::cout << "\n--- Results ---" << std::endl;
        std::cout << "  Workers: " << config.num_cpu_workers << std::endl;
        std::cout << "  Time (ms): " << std::fixed << std::setprecision(2) << elapsed_ms << std::endl;
        std::cout << "  Throughput (docs/s): " << std::fixed << std::setprecision(0) << throughput << std::endl;

        if (index_ofs)
        {
            index_ofs << config.num_cpu_workers << ","
                      << elapsed_ms << ","
                      << throughput << "\n";
        }

        fs::remove_all(Config::TEMP_PATH);
    }

    if (build_index_mode)
    {
        std::cout << "\n[INDEXING MODE - STREAMING]" << std::endl;
        print_separator('-');
        
        // NEW: Use DocumentStream instead of loading all documents
        DocumentStream doc_stream(Config::CORPUS_DIR);
        
        std::cout << "\nIndexing " << doc_stream.size() << " documents using streaming approach..." << std::endl;
        std::cout << "Memory usage will remain constant regardless of corpus size." << std::endl;
        
        Indexer indexer(
            doc_stream,
            Config::INDEX_PATH, 
            Config::TEMP_PATH, 
            Config::DEFAULT_BLOCK_SIZE_MB, 
            num_shards, 
            config.num_cpu_workers
        );
        
        indexer.build_index();
        std::cout << "\nSharded streaming indexing complete (" << num_shards << " shards)." << std::endl;
    }

    if (run_benchmark_mode)
    {
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
            auto candidates = system.search_boolean(query, config.print_log);

            std::cout << "\n--- Top 5 Pure Boolean Results (Unranked) ---" << std::endl;
            std::cout << "  Found " << candidates.size() << " total documents." << std::endl;

            for (size_t i = 0; i < std::min((size_t)5, candidates.size()); ++i)
            {
                const std::string *doc_name = doc_store.get_document_name(candidates[i].doc_id);
                if (doc_name)
                {
                    std::cout << "  " << (i + 1) << ". Document: " << "./" << Config::CORPUS_DIR << "/" << *doc_name << ".txt"
                              << " (ID: " << candidates[i].doc_id << ")" << std::endl;
                }
                else
                {
                    std::cout << "  " << (i + 1) << ". DocID: " << candidates[i].doc_id
                              << " (name unavailable)" << std::endl;
                }
            }

            std::cout << "\n(Taking top " << Config::MAX_RERANK_CANDIDATES << " candidates for reranking...)" << std::endl;

            std::vector<Document> docs_to_rerank;
            if (!candidates.empty())
            {
                size_t rerank_count = std::min(candidates.size(), (size_t)Config::MAX_RERANK_CANDIDATES);
                docs_to_rerank.reserve(rerank_count);

                for (size_t i = 0; i < rerank_count; ++i)
                {
                    const Document *doc_ptr = doc_store.get_document(candidates[i].doc_id);
                    if (doc_ptr)
                    {
                        docs_to_rerank.emplace_back(candidates[i].doc_id,
                                                    truncate_to_words(doc_ptr->content, Config::DOCUMENT_TRUNCATE_WORDS));
                    }
                }
            }

            auto reranked = gpu_reranker.rerank_with_chunking(query, docs_to_rerank);
            auto end_time = std::chrono::high_resolution_clock::now();
            double elapsed_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();

            std::cout << "\n--- Top 5 Neurally Reranked Results ---" << std::endl;

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