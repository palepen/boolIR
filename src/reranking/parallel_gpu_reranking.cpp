#include "reranking/parallel_gpu_reranking.h"
#include "system_controller.h"
#include "benchmark_suite.h"
#include <cilk/cilk.h>
#include <atomic>
#include <mutex>
#include <chrono>
#include <iostream>

static std::unordered_map<unsigned int, const Document*> build_doc_id_map(
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
    GpuRerankService& rerank_service,
    const std::vector<std::pair<std::string, std::string>>& queries,
    const DocumentCollection& documents,
    std::vector<QueryMetrics>& query_metrics
) {
    const size_t num_queries = queries.size();
    auto doc_id_map = build_doc_id_map(documents);
    
    std::vector<std::future<std::vector<ScoredDocument>>> rerank_futures(num_queries);
    query_metrics.resize(num_queries);

    // Producer stage: retrieve candidates in parallel
    cilk_for (size_t i = 0; i < num_queries; ++i) {
        const auto& [query_id, query_text] = queries[i];
        
        auto start_retrieval = std::chrono::high_resolution_clock::now();
        std::unique_ptr<QueryNode> query_tree = system.expand_query(query_text);
        ResultSet candidates = system.execute_boolean_query(*query_tree);
        auto end_retrieval = std::chrono::high_resolution_clock::now();
        
        // Apply simple heuristics to get ordered candidates
        std::vector<unsigned int> ordered_doc_ids;
        
        // Simple ordering: prefer documents earlier in the list
        // (In practice, you might want document length here)
        if (candidates.doc_ids.size() > 1000) {
            ordered_doc_ids.assign(candidates.doc_ids.begin(), 
                                  candidates.doc_ids.begin() + 1000);
        } else {
            ordered_doc_ids = candidates.doc_ids;
        }
        
        const size_t max_candidates_to_rerank = 1000;
        std::vector<Document> candidate_docs;
        size_t limit = std::min(max_candidates_to_rerank, ordered_doc_ids.size());
        candidate_docs.reserve(limit);
        
        for (size_t j = 0; j < limit; ++j) {
            auto it = doc_id_map.find(ordered_doc_ids[j]);
            if (it != doc_id_map.end()) {
                candidate_docs.push_back(*(it->second));
            }
        }

        query_metrics[i].query_id = query_id;
        query_metrics[i].num_candidates = candidates.doc_ids.size();
        query_metrics[i].retrieval_time_ms = std::chrono::duration<double, std::milli>(end_retrieval - start_retrieval).count();
        
        // Submit job only if we have candidates
        if (!candidate_docs.empty()) {
            rerank_futures[i] = rerank_service.submit_job(query_text, candidate_docs);
        }
    }

    // Consumer stage: collect reranked results
    std::vector<std::pair<std::string, std::vector<SearchResult>>> results_vec(num_queries);
    
    cilk_for(size_t i = 0; i < num_queries; ++i) {
        try {
            // Check if we have a valid future
            if (!rerank_futures[i].valid()) {
                results_vec[i] = {queries[i].first, {}};
                continue;
            }
            
            auto start_wait = std::chrono::high_resolution_clock::now();
            auto reranked_docs = rerank_futures[i].get();
            auto end_wait = std::chrono::high_resolution_clock::now();
            
            query_metrics[i].reranking_time_ms = std::chrono::duration<double, std::milli>(end_wait - start_wait).count();

            std::vector<SearchResult> final_results;
            final_results.reserve(reranked_docs.size());
            for (const auto& doc : reranked_docs) {
                final_results.push_back({doc.id, doc.score});
            }
            results_vec[i] = {queries[i].first, final_results};

        } catch (const std::exception& e) {
            std::cerr << "Error processing query " << queries[i].first << ": " << e.what() << std::endl;
            results_vec[i] = {queries[i].first, {}};
        } catch (...) {
            std::cerr << "Unknown error processing query " << queries[i].first << std::endl;
            results_vec[i] = {queries[i].first, {}};
        }
    }
    
    std::unordered_map<std::string, std::vector<SearchResult>> results_map;
    for (const auto& pair : results_vec) {
        results_map[pair.first] = pair.second;
    }
    
    return results_map;
}