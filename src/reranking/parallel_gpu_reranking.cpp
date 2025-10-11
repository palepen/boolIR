#include "reranking/parallel_gpu_reranking.h"
#include "system_controller.h"
#include <cilk/cilk.h>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <chrono>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <unordered_map>
#include "retrieval/query.h"

// Parse query string into QueryNode tree
QueryNode parse_query(const std::string& query_str) {
    QueryNode query_tree(QueryOperator::AND);
    std::stringstream ss(query_str);
    std::string term;
    
    while (ss >> term) {
        for (char& c : term) {
            c = tolower(static_cast<unsigned char>(c));
        }
        query_tree.children.push_back(std::make_unique<QueryNode>(term));
    }
    
    return query_tree;
}

BatchedGpuReranker::BatchedGpuReranker(const char* model_path, const char* vocab_path) 
    : reranker_(std::make_unique<GpuNeuralReranker>(model_path, vocab_path, 32)), 
      stop_processing_(false) {
    gpu_worker_ = std::thread(&BatchedGpuReranker::process_batches, this);
}

BatchedGpuReranker::~BatchedGpuReranker() {
    stop_processing_ = true;
    queue_cv_.notify_all();
    if (gpu_worker_.joinable()) {
        gpu_worker_.join();
    }
}

std::future<std::vector<ScoredDocument>> BatchedGpuReranker::submit_query(
    const std::string& query_id,
    const std::string& query_text,
    const std::vector<Document>& docs
) {
    std::promise<std::vector<ScoredDocument>> promise;
    auto future = promise.get_future();
    
    // Handle empty candidate sets immediately
    if (docs.empty()) {
        std::cout << "  [WARNING] Query " << query_id << " has no candidates" << std::endl;
        promise.set_value({});
        return future;
    }
    
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        if (!batch_queue_.empty() && 
            batch_queue_.back().queries.size() < MAX_BATCH_SIZE) {
            auto& batch = batch_queue_.back();
            batch.queries.push_back({query_id, query_text});
            batch.documents.push_back(docs);
            batch.promises.push_back(std::move(promise));
        } else {
            QueryBatch batch;
            batch.queries.push_back({query_id, query_text});
            batch.documents.push_back(docs);
            batch.promises.push_back(std::move(promise));
            batch_queue_.push(std::move(batch));
        }
    }
    
    queue_cv_.notify_one();
    return future;
}

void BatchedGpuReranker::process_batches() {
    while (!stop_processing_) {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        
        queue_cv_.wait_for(lock, std::chrono::milliseconds(MAX_WAIT_MS),
            [this] { 
                return !batch_queue_.empty() || stop_processing_;
            });
        
        if (!batch_queue_.empty()) {
            QueryBatch batch = std::move(batch_queue_.front());
            batch_queue_.pop();
            lock.unlock();
            
            process_batch_on_gpu(batch);
        }
    }
}

void BatchedGpuReranker::process_batch_on_gpu(QueryBatch& batch) {
    try {
        auto start = std::chrono::high_resolution_clock::now();
        
        std::cout << "  [GPU BATCH] Processing " << batch.queries.size() << " queries..." << std::endl;
        
        // Process each query in the batch
        for (size_t i = 0; i < batch.queries.size(); i++) {
            const auto& [qid, qtext] = batch.queries[i];
            const auto& docs = batch.documents[i];
            
            if (docs.empty()) {
                std::cout << "  [WARNING] Query " << qid << " has empty doc set" << std::endl;
                batch.promises[i].set_value({});
                continue;
            }
            
            try {
                // CRITICAL: Use the reranker's built-in rerank() method
                // This ensures proper semantic scoring
                auto results = reranker_->rerank(qtext, docs);
                
                // Validate results
                if (results.empty() && !docs.empty()) {
                    std::cerr << "  [ERROR] Query " << qid << " produced no results from " 
                              << docs.size() << " candidates" << std::endl;
                }
                
                // Fulfill promise
                batch.promises[i].set_value(std::move(results));
                
                std::cout << "  [SUCCESS] Query " << qid << " reranked " 
                          << results.size() << " documents" << std::endl;
                
            } catch (const std::exception& e) {
                std::cerr << "  [ERROR] Query " << qid << " failed: " << e.what() << std::endl;
                batch.promises[i].set_value({});
            }
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            end - start).count();
        
        std::cout << "  [GPU BATCH] Completed " << batch.queries.size() 
                  << " queries in " << duration << " ms" << std::endl;
                  
    } catch (const std::exception& e) {
        std::cerr << "  [FATAL] Batch processing error: " << e.what() << std::endl;
        for (auto& promise : batch.promises) {
            try {
                promise.set_value({});
            } catch (...) {}
        }
    }
}

// CRITICAL FIX: Build document ID mapping for O(1) lookup
std::unordered_map<unsigned int, const Document*> build_doc_id_map(
    const DocumentCollection& documents
) {
    std::unordered_map<unsigned int, const Document*> doc_map;
    doc_map.reserve(documents.size());
    
    for (const auto& doc : documents) {
        doc_map[doc.id] = &doc;
    }
    
    return doc_map;
}

std::unordered_map<std::string, std::vector<SearchResult>> 
process_queries_parallel_batched(
    HighPerformanceIRSystem& system,
    BatchedGpuReranker& gpu_reranker,
    const std::vector<std::pair<std::string, std::string>>& queries,
    const DocumentCollection& documents,
    std::vector<QueryMetrics>& query_metrics
) {
    const size_t num_queries = queries.size();
    std::cout << "\n========================================" << std::endl;
    std::cout << "PARALLEL BATCHED RERANKING" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "  Total queries: " << num_queries << std::endl;
    std::cout << "  Document collection size: " << documents.size() << std::endl;
    
    // CRITICAL OPTIMIZATION: Build document ID mapping once
    std::cout << "  Building document ID map..." << std::endl;
    auto doc_id_map = build_doc_id_map(documents);
    std::cout << "  Document ID map built with " << doc_id_map.size() << " entries" << std::endl;
    
    std::vector<std::pair<std::string, std::vector<SearchResult>>> results_vec(num_queries);
    std::vector<QueryMetrics> metrics_vec(num_queries);
    
    std::atomic<size_t> completed_retrieval(0);
    std::atomic<size_t> completed_reranking(0);
    std::mutex progress_mutex;
    
    auto start_total = std::chrono::high_resolution_clock::now();
    
    // ========================================================================
    // PHASE 1: Parallel Boolean Retrieval with Cilk
    // ========================================================================
    std::cout << "\n[PHASE 1] Boolean Retrieval..." << std::endl;
    std::vector<std::vector<Document>> all_candidate_docs(num_queries);
    
    cilk_for(size_t i = 0; i < num_queries; ++i) {
        const auto& [query_id, query_text] = queries[i];
        
        auto start_retrieval = std::chrono::high_resolution_clock::now();
        
        // Execute boolean query
        QueryNode query_tree = parse_query(query_text);
        ResultSet candidates = system.execute_boolean_query(query_tree);
        
        auto end_retrieval = std::chrono::high_resolution_clock::now();
        auto retrieval_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_retrieval - start_retrieval).count();
        
        // CRITICAL FIX: Use document ID map for O(1) lookup
        std::vector<Document> candidate_docs;
        size_t max_candidates = std::min(size_t(100), candidates.doc_ids.size());
        candidate_docs.reserve(max_candidates);
        
        for (size_t j = 0; j < max_candidates && j < candidates.doc_ids.size(); ++j) {
            unsigned int doc_id = candidates.doc_ids[j];
            auto it = doc_id_map.find(doc_id);
            if (it != doc_id_map.end()) {
                candidate_docs.push_back(*(it->second));
            } else {
                std::cerr << "  [WARNING] Document ID " << doc_id << " not found in map" << std::endl;
            }
        }
        
        all_candidate_docs[i] = candidate_docs;
        
        // Store metadata
        results_vec[i].first = query_id;
        metrics_vec[i].query_id = query_id;
        metrics_vec[i].num_candidates = candidates.doc_ids.size();
        metrics_vec[i].retrieval_time_ms = retrieval_ms;
        
        size_t completed = completed_retrieval.fetch_add(1) + 1;
        if (completed % 10 == 0 || completed == num_queries) {
            std::lock_guard<std::mutex> lock(progress_mutex);
            std::cout << "  [RETRIEVAL] " << completed << "/" << num_queries 
                      << " queries completed" << std::endl;
        }
    }
    
    std::cout << "[PHASE 1] Boolean retrieval completed" << std::endl;
    
    // ========================================================================
    // PHASE 2: Submit all queries to GPU reranker
    // ========================================================================
    std::cout << "\n[PHASE 2] Submitting to GPU reranker..." << std::endl;
    std::vector<std::future<std::vector<ScoredDocument>>> rerank_futures;
    rerank_futures.reserve(num_queries);
    
    for (size_t i = 0; i < num_queries; ++i) {
        const auto& [query_id, query_text] = queries[i];
        
        if (all_candidate_docs[i].empty()) {
            std::cout << "  [WARNING] Query " << query_id << " has no candidates to rerank" << std::endl;
            // Create a dummy future that returns empty results
            std::promise<std::vector<ScoredDocument>> empty_promise;
            empty_promise.set_value({});
            rerank_futures.push_back(empty_promise.get_future());
        } else {
            auto future = gpu_reranker.submit_query(query_id, query_text, all_candidate_docs[i]);
            rerank_futures.push_back(std::move(future));
        }
    }
    
    std::cout << "[PHASE 2] All queries submitted" << std::endl;
    
    // ========================================================================
    // PHASE 3: Collect reranking results in parallel
    // ========================================================================
    std::cout << "\n[PHASE 3] Collecting reranking results..." << std::endl;
    
    cilk_for(size_t i = 0; i < num_queries; ++i) {
        auto start_wait = std::chrono::high_resolution_clock::now();
        
        try {
            // Wait for GPU reranking result
            auto reranked = rerank_futures[i].get();
            
            auto end_wait = std::chrono::high_resolution_clock::now();
            auto rerank_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                end_wait - start_wait).count();
            
            // Validate results
            if (reranked.empty() && !all_candidate_docs[i].empty()) {
                std::cerr << "  [WARNING] Query " << results_vec[i].first 
                          << " returned empty results from " << all_candidate_docs[i].size() 
                          << " candidates" << std::endl;
            }
            
            // Convert to SearchResult format
            std::vector<SearchResult> final_results;
            final_results.reserve(reranked.size());
            for (const auto& scored : reranked) {
                final_results.push_back(SearchResult{scored.id, scored.score});
            }
            
            results_vec[i].second = std::move(final_results);
            metrics_vec[i].reranking_time_ms = rerank_ms;
            
            size_t completed = completed_reranking.fetch_add(1) + 1;
            if (completed % 10 == 0 || completed == num_queries) {
                std::lock_guard<std::mutex> lock(progress_mutex);
                std::cout << "  [RERANKING] " << completed << "/" << num_queries 
                          << " queries completed" << std::endl;
            }
        } catch (const std::exception& e) {
            std::cerr << "  [ERROR] Query " << results_vec[i].first 
                      << " reranking failed: " << e.what() << std::endl;
            results_vec[i].second = {};
            metrics_vec[i].reranking_time_ms = 0;
        }
    }
    
    auto end_total = std::chrono::high_resolution_clock::now();
    auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_total - start_total).count();
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "PARALLEL BATCHED RERANKING COMPLETE" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "  Total time: " << total_ms << " ms" << std::endl;
    std::cout << "  Throughput: " << std::fixed << std::setprecision(2)
              << (1000.0 * num_queries / total_ms) << " queries/sec" << std::endl;
    
    // Statistics
    size_t total_candidates = 0;
    size_t total_reranked = 0;
    for (size_t i = 0; i < num_queries; ++i) {
        total_candidates += metrics_vec[i].num_candidates;
        total_reranked += results_vec[i].second.size();
    }
    std::cout << "  Total candidates: " << total_candidates << std::endl;
    std::cout << "  Total reranked: " << total_reranked << std::endl;
    std::cout << "  Avg candidates/query: " << (total_candidates / num_queries) << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    // Convert to map and return
    std::unordered_map<std::string, std::vector<SearchResult>> results_map;
    results_map.reserve(num_queries);
    
    for (const auto& [qid, res] : results_vec) {
        results_map[qid] = res;
    }
    query_metrics = metrics_vec;
    
    return results_map;
}