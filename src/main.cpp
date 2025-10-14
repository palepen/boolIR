#include "system_controller.h"
#include "benchmark_suite.h"
#include "data_loader.h"
#include "retrieval/pre_ranker.h"
#include "reranking/neural_reranker.h"
#include "indexing/bsbi_indexer.h"
#include <iostream>
#include <string>
#include <filesystem>
#include <chrono>
#include <thread>
#include <algorithm>
#include <sstream>

namespace fs = std::filesystem;

void print_separator(char ch = '=', int width = 80) {
    std::cout << std::string(width, ch) << std::endl;
}

// Helper function to truncate document content to first N words
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

int main(int argc, char** argv) {
    print_separator();
    std::cout << "High-Performance IR System" << std::endl;
    print_separator();

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
    bool run_demo_mode = false;
    bool benchmark_indexing = false;

    // Command-line argument parsing
    BenchmarkConfig config;
    config.use_reranking = true;
    config.num_cpu_workers = std::thread::hardware_concurrency();
    config.label = "default_run";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--build-index") {
            build_index_mode = true;
        } else if (arg == "--benchmark") {
            run_benchmark_mode = true;
        } else if (arg == "--demo") {
            run_demo_mode = true;
        } else if (arg == "--benchmark-indexing") {
            benchmark_indexing = true;
        } else if (arg == "--label" && i + 1 < argc) {
            config.label = argv[++i];
        } else if (arg == "--cpu-workers" && i + 1 < argc) {
            config.num_cpu_workers = std::stoi(argv[++i]);
        } else if (arg == "--no-rerank") {
            config.use_reranking = false;
        }
    }
    
    // Default to demo mode if no other mode is specified
    if (!build_index_mode && !run_benchmark_mode && !run_demo_mode && !benchmark_indexing) {
        run_demo_mode = true;
    }

    if (benchmark_indexing) {
        std::cout << "\n[INDEXING BENCHMARK MODE]" << std::endl;
        print_separator('-');
        
        auto [documents, doc_name_to_id] = load_trec_documents(corpus_dir);
        
        std::vector<size_t> worker_counts = {1, 2, 4, 8};
        std::vector<double> indexing_times;
        std::vector<double> throughputs;
        
        std::cout << "\n=== INDEXING SCALABILITY BENCHMARK ===" << std::endl;
        std::cout << "Total documents: " << documents.size() << std::endl;
        std::cout << "\nTesting with different worker counts...\n" << std::endl;
        
        for (size_t workers : worker_counts) {
            std::string test_index_path = index_path + "_test_" + std::to_string(workers);
            std::string test_temp_path = temp_path + "_test_" + std::to_string(workers);
            
            // Clean up previous test runs
            fs::remove_all(test_index_path);
            fs::remove_all(test_temp_path);
            
            std::cout << "--- Testing with " << workers << " CPU workers ---" << std::endl;
            
            // Set Cilk workers
            std::string cilk_env = "CILK_NWORKERS=" + std::to_string(workers);
            putenv(const_cast<char*>(cilk_env.c_str()));
            
            BSBIIndexer indexer(documents, test_index_path, test_temp_path);
            
            auto start = std::chrono::high_resolution_clock::now();
            indexer.build_index();
            auto end = std::chrono::high_resolution_clock::now();
            
            double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
            double throughput = (documents.size() * 1000.0) / elapsed_ms;
            
            indexing_times.push_back(elapsed_ms);
            throughputs.push_back(throughput);
            
            std::cout << "  Total time: " << std::fixed << std::setprecision(2) 
                      << elapsed_ms << " ms" << std::endl;
            std::cout << "  Throughput: " << std::fixed << std::setprecision(0) 
                      << throughput << " docs/sec" << std::endl;
            
            // Clean up test directories
            fs::remove_all(test_index_path);
            fs::remove_all(test_temp_path);
        }
        
        std::cout << "\n=== INDEXING SCALABILITY SUMMARY ===" << std::endl;
        std::cout << "Workers | Time (ms) | Throughput (docs/s) | Speedup | Efficiency" << std::endl;
        std::cout << "--------|-----------|---------------------|---------|------------" << std::endl;
        
        double baseline_time = indexing_times[0];
        for (size_t i = 0; i < worker_counts.size(); ++i) {
            double speedup = baseline_time / indexing_times[i];
            double efficiency = (speedup / worker_counts[i]) * 100.0;
            
            std::cout << std::setw(7) << worker_counts[i] << " | "
                      << std::setw(9) << std::fixed << std::setprecision(0) << indexing_times[i] << " | "
                      << std::setw(19) << std::fixed << std::setprecision(0) << throughputs[i] << " | "
                      << std::setw(7) << std::fixed << std::setprecision(2) << speedup << " | "
                      << std::setw(10) << std::fixed << std::setprecision(1) << efficiency << "%" << std::endl;
        }
        
        std::cout << "\n";
    }

    if (build_index_mode) {
        std::cout << "\n[INDEXING MODE]" << std::endl;
        print_separator('-');
        
        auto [documents, doc_name_to_id] = load_trec_documents(corpus_dir);
        BSBIIndexer indexer(documents, index_path, temp_path);
        indexer.build_index();
        
        std::cout << "\nIndexing complete. Persistent index created in '" << index_path << "' directory." << std::endl;
    }
    
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
            
            // Run Boolean benchmark first
            BenchmarkConfig bool_config = config;
            bool_config.use_reranking = false;
            bool_config.label = "Boolean_" + std::to_string(config.num_cpu_workers) + "-cpu";
            suite.run_single_benchmark(bool_config);
            
            // Then run Reranking benchmark if requested
            if (config.use_reranking) {
                BenchmarkConfig rerank_config = config;
                rerank_config.use_reranking = true;
                rerank_config.label = "Rerank_" + std::to_string(config.num_cpu_workers) + "-cpu";
                suite.run_single_benchmark(rerank_config);
                
                // Print detailed comparison
                suite.print_comparison();
            }
        }

        if (run_demo_mode) {
            std::cout << "\nRunning Quick Demo with Neural Reranking..." << std::endl;
            std::cout << "OPTIMIZATIONS ENABLED:" << std::endl;
            std::cout << "  - Document truncation: 200 words" << std::endl;
            std::cout << "  - Max sequence length: 256 tokens" << std::endl;
            std::cout << "  - Batch size: 32" << std::endl;
            std::cout << "  - Top-K reranking: 100 candidates (rest preserved)" << std::endl;
            
            // Create a map for efficient document lookup
            std::unordered_map<unsigned int, const Document*> doc_id_map;
            for(const auto& doc : documents) {
                doc_id_map[doc.id] = &doc;
            }

            HighPerformanceIRSystem system(index_path, synonym_path, std::make_unique<TermOverlapRanker>());
            GpuNeuralReranker gpu_reranker(model_path.c_str(), vocab_path.c_str(), 32);

            int count = 0;
            for (const auto& [qid, qtext] : topics) {
                if (++count > 5) break;
                std::cout << "\n--- Query: \"" << qtext << "\" ---" << std::endl;
                
                auto candidates = system.search_boolean(qtext, documents);
                std::cout << "  Boolean retrieval: " << candidates.size() << " candidates" << std::endl;
                
                // Rerank top 100, keep rest
                const size_t MAX_RERANK = 100;
                std::vector<SearchResult> final_results;
                
                if (candidates.size() <= MAX_RERANK) {
                    std::vector<Document> candidate_docs;
                    candidate_docs.reserve(candidates.size());
                    for (const auto& res : candidates) {
                        auto it = doc_id_map.find(res.doc_id);
                        if (it != doc_id_map.end()) {
                            std::string truncated_content = truncate_to_words(it->second->content, 200);
                            candidate_docs.emplace_back(res.doc_id, truncated_content);
                        }
                    }

                    auto reranked_scored_docs = gpu_reranker.rerank_with_chunking(qtext, candidate_docs);
                    for(const auto& sd : reranked_scored_docs) {
                        final_results.push_back({sd.id, sd.score});
                    }
                } else {
                    std::vector<Document> top_k_docs;
                    top_k_docs.reserve(MAX_RERANK);
                    for (size_t j = 0; j < MAX_RERANK; ++j) {
                        auto it = doc_id_map.find(candidates[j].doc_id);
                        if (it != doc_id_map.end()) {
                            std::string truncated_content = truncate_to_words(it->second->content, 200);
                            top_k_docs.emplace_back(candidates[j].doc_id, truncated_content);
                        }
                    }

                    auto reranked_scored_docs = gpu_reranker.rerank_with_chunking(qtext, top_k_docs);
                    
                    for(const auto& sd : reranked_scored_docs) {
                        final_results.push_back({sd.id, sd.score});
                    }
                    
                    // Append remaining with decayed scores
                    float min_score = final_results.empty() ? 0.0f : final_results.back().score;
                    for (size_t j = MAX_RERANK; j < candidates.size(); ++j) {
                        float decayed = min_score * 0.9f * (1.0f - (j - MAX_RERANK) * 0.0001f);
                        final_results.push_back({candidates[j].doc_id, decayed});
                    }
                    
                    std::cout << "  Reranked top " << MAX_RERANK << ", preserved " 
                              << (candidates.size() - MAX_RERANK) << " remaining" << std::endl;
                }

                std::cout << "  -> Top 3 results:" << std::endl;
                for (size_t i = 0; i < std::min((size_t)3, final_results.size()); ++i) {
                     std::cout << "     " << (i+1) << ". DocID: " << final_results[i].doc_id 
                               << " (Score: " << std::fixed << std::setprecision(4) 
                               << final_results[i].score << ")" << std::endl;
                }
            }
        }
    }

    print_separator();
    std::cout << "ALL OPERATIONS COMPLETED SUCCESSFULLY!" << std::endl;
    print_separator();
    
    return 0;
}