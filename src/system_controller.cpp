#include "system_controller.h"
#include <iostream>
#include <algorithm>
#include <stdexcept>
#include <sstream> // Required for std::stringstream

HighPerformanceIRSystem::HighPerformanceIRSystem(size_t num_shards, const char* model_path)
    : indexer_(num_shards) {
    // Retriever and Reranker can only be initialized after an index is built
    // or loaded. We will create them after build_index.
    // The reranker is initialized here to load the model.
    try {
        reranker_ = std::make_unique<GpuNeuralReranker>(model_path, 32);
    } catch (const std::exception& e) {
        std::cerr << "Could not initialize GPU Reranker: " << e.what() << std::endl;
        std::cerr << "Falling back to CPU-only operations might be necessary." << std::endl;
    }
}

void HighPerformanceIRSystem::build_index(const DocumentCollection& documents) {
    std::cout << "--- Building Index for the Full System ---" << std::endl;
    documents_ptr_ = &documents; // Store pointer to documents
    indexer_.build_index_parallel(documents);
    retriever_ = std::make_unique<ParallelBooleanRetrieval>(indexer_.get_full_index());
    std::cout << "--- Index built and components are ready ---" << std::endl;
}

std::vector<SearchResult> HighPerformanceIRSystem::search(const std::string& query_str, bool use_reranking) {
    if (!retriever_) {
        throw std::runtime_error("Index has not been built. Please call build_index() before searching.");
    }

    // --- Stage 1: Parallel Boolean Retrieval ---
    // For this example, we'll parse a simple AND query from the string
    QueryNode query_tree(QueryOperator::AND);
    std::stringstream ss(query_str);
    std::string term;
    while (ss >> term) {
        for(char& c : term) c = tolower(c);
        query_tree.children.push_back(std::make_unique<QueryNode>(term));
    }
    
    ResultSet candidates_result = retriever_->execute_query(query_tree);
    const auto& candidate_ids = candidates_result.doc_ids;

    std::cout << "\nStage 1 (Boolean Retrieval) found " << candidate_ids.size() << " candidates for query: '" << query_str << "'" << std::endl;

    if (!use_reranking || !reranker_) {
        std::vector<SearchResult> final_results;
        for (unsigned int id : candidate_ids) {
            // Assign a dummy score of 1.0 for non-reranked results
            // MODIFIED: Explicitly construct the SearchResult object
            final_results.push_back(SearchResult{id, 1.0f});
        }
        return final_results;
    }

    // --- Stage 2: GPU Neural Re-ranking ---
    // Prepare candidate documents for the reranker
    std::vector<Document> candidate_docs;
    for (unsigned int id : candidate_ids) {
        // This is a slow lookup; a real system would use a doc store (e.g., a map)
        for (const auto& doc : *documents_ptr_) {
            if (doc.id == id) {
                candidate_docs.push_back(doc);
                break;
            }
        }
    }

    std::cout << "Stage 2 (Neural Re-ranking) is processing " << candidate_docs.size() << " candidates..." << std::endl;
    std::vector<ScoredDocument> reranked_docs = reranker_->rerank(query_str, candidate_docs);

    // Convert to final SearchResult format
    std::vector<SearchResult> final_results;
    for (const auto& scored_doc : reranked_docs) {
        final_results.push_back({scored_doc.id, scored_doc.score});
    }

    // The reranker already sorts, but we can ensure it here as well.
    std::sort(final_results.begin(), final_results.end());

    return final_results;
}