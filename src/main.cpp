#include "indexing/bsbi_indexer.h"
#include "system_controller.h"
#include "benchmark_suite.h"
#include "data_loader.h"
#include <iostream>
#include <string>
#include <filesystem>
#include <chrono>

namespace fs = std::filesystem;

void print_separator(char ch = '=', int width = 80) {
    std::cout << std::string(width, ch) << std::endl;
}

int main(int argc, char** argv) {
    print_separator();
    std::cout << "High-Performance IR System" << std::endl;
    print_separator();

    // --- Configuration ---
    const std::string corpus_dir = "data/cord19-trec-covid_corpus_batched";
    const std::string topics_path = "data/topics.cord19-trec-covid.txt";
    const std::string qrels_path = "data/qrels.cord19-trec-covid.txt";
    const std::string synonym_path = "data/synonyms.txt";
    const std::string model_path = "models/bert_model.onnx";
    const std::string vocab_path = "models/vocab.txt";
    const std::string index_path = "index";
    const std::string temp_path = "index/temp";

    bool build_index_mode = false;
    bool run_benchmark_mode = false;
    bool run_demo_mode = true; 

    if (argc > 1) {
        std::string arg = argv[1];
        if (arg == "--build-index") {
            build_index_mode = true;
            run_demo_mode = false;
        } else if (arg == "--benchmark") {
            run_benchmark_mode = true;
            run_demo_mode = false;
        } else if (arg == "--demo") {
            run_demo_mode = true;
        }
    }

    // --- Mode 1: Build Index ---
    if (build_index_mode) {
        std::cout << "\n[INDEXING MODE]" << std::endl;
        print_separator('-');
        
        auto [documents, doc_name_to_id] = load_trec_documents(corpus_dir);
        BSBIIndexer indexer(documents, index_path, temp_path);
        indexer.build_index();
        
        std::cout << "\nIndexing complete. Persistent index created in '" << index_path << "' directory." << std::endl;
    }
    
    // --- Mode 2: Run Benchmarks / Demo ---
    if (run_benchmark_mode || run_demo_mode) {
        if (!fs::exists(index_path + "/dictionary.dat")) {
            std::cerr << "FATAL: Index not found. Please run with '--build-index' first." << std::endl;
            return 1;
        }
        
        std::cout << "\n[SEARCH MODE]" << std::endl;
        print_separator('-');

        auto [documents, doc_name_to_id] = load_trec_documents(corpus_dir);
        auto topics = load_trec_topics(topics_path);
        Qrels ground_truth = load_trec_qrels(qrels_path, doc_name_to_id);

        if (run_benchmark_mode) {
            BenchmarkSuite suite(documents, topics, ground_truth, model_path, vocab_path, index_path, synonym_path);
            suite.run_full_benchmark();
        }

        if (run_demo_mode) {
            std::cout << "\nRunning Quick Demo..." << std::endl;
            HighPerformanceIRSystem system(index_path, synonym_path);
            GpuNeuralReranker reranker(model_path.c_str(), vocab_path.c_str());

            int count = 0;
            for (const auto& [qid, qtext] : topics) {
                if (++count > 5) break;
                std::cout << "\n--- Query: \"" << qtext << "\" ---" << std::endl;
                
                // **THE FIX IS HERE**:
                // The boolean flag `true` is removed to match the correct 3-argument 'search' method.
                auto results = system.search(qtext, reranker, documents);

                std::cout << "  -> Top 3 results:" << std::endl;
                for (size_t i = 0; i < std::min((size_t)3, results.size()); ++i) {
                     std::cout << "     DocID: " << results[i].doc_id << " (Score: " << results[i].score << ")" << std::endl;
                }
            }
        }
    }

    print_separator();
    std::cout << "ALL OPERATIONS COMPLETED SUCCESSFULLY!" << std::endl;
    print_separator();
    
    return 0;
}

