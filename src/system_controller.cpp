#include "system_controller.h"
#include <iostream>
#include <algorithm>
#include <stdexcept>
#include <sstream>
#include <iomanip>

HighPerformanceIRSystem::HighPerformanceIRSystem(
    size_t num_shards,
    const char *model_path,
    const char *vocab_path) : indexer_(num_shards)
{
    try
    {
        std::cout << "Initializing Batched GPU Neural Reranker..." << std::endl;
        reranker_ = std::make_unique<BatchedGpuReranker>(model_path, vocab_path);
        std::cout << "✓ Batched GPU Reranker initialized successfully" << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "⚠ Warning: Could not initialize GPU Reranker: " << e.what() << std::endl;
        std::cerr << "  System will operate without neural reranking." << std::endl;
    }
}

void HighPerformanceIRSystem::build_index(const DocumentCollection &documents)
{
    std::cout << "\n--- Building Inverted Index ---" << std::endl;
    documents_ptr_ = &documents;
    
    // CRITICAL: Build document ID mapping for O(1) lookup
    std::cout << "Building document ID mapping..." << std::endl;
    doc_id_map_.clear();
    doc_id_map_.reserve(documents.size());
    
    for (const auto& doc : documents) {
        doc_id_map_[doc.id] = &doc;
    }
    
    std::cout << "✓ Document ID map built: " << doc_id_map_.size() << " documents" << std::endl;

    // Build the parallel inverted index
    indexer_.build_index_parallel(documents);

    // Initialize the retriever with the built index
    retriever_ = std::make_unique<ParallelRetrieval>(indexer_.get_full_index());

    std::cout << "✓ Index built successfully" << std::endl;
    std::cout << "  • Total documents indexed: " << documents.size() << std::endl;

    // Print indexing performance metrics
    auto metrics = indexer_.get_performance_metrics();
    std::cout << "  • Indexing time: " << std::fixed << std::setprecision(2)
              << metrics.indexing_time_ms << " ms" << std::endl;
    std::cout << "  • Throughput: " << std::fixed << std::setprecision(2)
              << metrics.throughput_docs_per_sec << " docs/sec" << std::endl;
}

std::vector<SearchResult> HighPerformanceIRSystem::search(
    const std::string &query_str,
    bool use_reranking)
{
    if (!retriever_)
    {
        throw std::runtime_error("Index has not been built. Please call build_index() before searching.");
    }

    // ========================================================================
    // STAGE 1: Boolean Retrieval (Fast Candidate Selection)
    // ========================================================================
    
    auto start_retrieval = std::chrono::high_resolution_clock::now();
    
    // Parse query into boolean query tree
    QueryNode query_tree(QueryOperator::AND);
    std::stringstream ss(query_str);
    std::string term;

    while (ss >> term)
    {
        for (char &c : term)
        {
            c = tolower(static_cast<unsigned char>(c));
        }
        query_tree.children.push_back(std::make_unique<QueryNode>(term));
    }

    // Execute boolean retrieval
    ResultSet candidates_result = retriever_->execute_query_optimized(query_tree);
    const auto &candidate_ids = candidates_result.doc_ids;
    
    auto end_retrieval = std::chrono::high_resolution_clock::now();
    auto retrieval_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_retrieval - start_retrieval).count();

    std::cout << "  Stage 1 (Boolean Retrieval): Found " << candidate_ids.size()
              << " candidates in " << retrieval_ms << " ms" << std::endl;

    // If no reranking requested or reranker not available
    if (!use_reranking || !reranker_)
    {
        std::vector<SearchResult> final_results;
        final_results.reserve(candidate_ids.size());

        for (unsigned int id : candidate_ids)
        {
            final_results.push_back(SearchResult{id, 1.0f});
        }

        if (!use_reranking)
        {
            std::cout << "  Stage 2 (Neural Reranking): Skipped (disabled)" << std::endl;
        }
        else
        {
            std::cout << "  Stage 2 (Neural Reranking): Skipped (reranker not available)" << std::endl;
        }

        return final_results;
    }

    // ========================================================================
    // STAGE 2: Neural Reranking (Semantic Relevance Scoring)
    // ========================================================================
    
    std::cout << "  Stage 2 (Neural Reranking): Processing " << candidate_ids.size() 
              << " candidates..." << std::endl;
    
    auto start_rerank = std::chrono::high_resolution_clock::now();
    
    // CRITICAL FIX: Use document ID map for O(1) lookup instead of O(n) linear search
    std::vector<Document> candidate_docs;
    size_t max_candidates = std::min(size_t(100), candidate_ids.size());
    candidate_docs.reserve(max_candidates);
    
    size_t found = 0;
    for (unsigned int id : candidate_ids) {
        if (found >= max_candidates) break;
        
        auto it = doc_id_map_.find(id);
        if (it != doc_id_map_.end()) {
            candidate_docs.push_back(*(it->second));
            found++;
        } else {
            std::cerr << "  [WARNING] Document ID " << id << " not found in map" << std::endl;
        }
    }
    
    if (candidate_docs.empty()) {
        std::cerr << "  [ERROR] No candidate documents found for reranking" << std::endl;
        return {};
    }

    std::cout << "  Prepared " << candidate_docs.size() << " documents for reranking" << std::endl;

    // Submit to GPU reranker
    auto future = reranker_->submit_query("single_query", query_str, candidate_docs);
    
    // Wait for results
    std::vector<ScoredDocument> reranked_docs;
    try {
        reranked_docs = future.get();
    } catch (const std::exception& e) {
        std::cerr << "  [ERROR] Reranking failed: " << e.what() << std::endl;
        // Fall back to boolean results
        std::vector<SearchResult> fallback_results;
        for (const auto& doc : candidate_docs) {
            fallback_results.push_back(SearchResult{doc.id, 1.0f});
        }
        return fallback_results;
    }
    
    auto end_rerank = std::chrono::high_resolution_clock::now();
    auto rerank_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_rerank - start_rerank).count();
    
    std::cout << "  Reranking completed in " << rerank_ms << " ms" << std::endl;

    // Validate results
    if (reranked_docs.empty()) {
        std::cerr << "  [WARNING] Reranker returned empty results" << std::endl;
        // Fall back to boolean results
        std::vector<SearchResult> fallback_results;
        for (const auto& doc : candidate_docs) {
            fallback_results.push_back(SearchResult{doc.id, 1.0f});
        }
        return fallback_results;
    }

    // Convert to SearchResult format
    std::vector<SearchResult> final_results;
    final_results.reserve(reranked_docs.size());

    for (const auto &scored_doc : reranked_docs)
    {
        final_results.push_back(SearchResult{scored_doc.id, scored_doc.score});
    }

    // Already sorted by reranker, but ensure it
    std::sort(final_results.begin(), final_results.end());

    std::cout << "  ✓ Reranking complete: " << final_results.size() << " documents scored" << std::endl;
    
    // Log score distribution
    if (!final_results.empty()) {
        std::cout << "  Score range: [" << final_results.back().score << ", " 
                  << final_results[0].score << "]" << std::endl;
    }

    return final_results;
}

ResultSet HighPerformanceIRSystem::execute_boolean_query(const QueryNode &query)
{
    if (!retriever_)
    {
        throw std::runtime_error("Index has not been built. Please call build_index() before searching.");
    }
    return retriever_->execute_query_optimized(query);
}