#include "indexing/parallel_indexer.h"
#include <cilk/cilk.h>
#include <sstream>
#include <iostream>

// Simple tokenizer used by the parallel indexer.
static std::vector<std::string> tokenize_parallel(const std::string& text) {
    std::vector<std::string> tokens;
    std::stringstream ss(text);
    std::string token;
    while (ss >> token) {
        for (char &c : token) {
            c = tolower(c);
        }
        tokens.push_back(token);
    }
    return tokens;
}

ParallelIndexer::ParallelIndexer(size_t num_shards) : shards_(num_shards) {}

size_t ParallelIndexer::hash_term_to_shard(const std::string& term) const {
    // std::hash provides a simple and effective hashing function.
    return std::hash<std::string>{}(term) % shards_.size();
}

void ParallelIndexer::build_index_parallel(const DocumentCollection& documents) {
    std::cout << "Starting parallel indexing for " << documents.size() << " documents with "
              << shards_.size() << " shards...\n";
    perf_monitor_.start_timer("total_indexing_time");

    // MAP Phase: Process documents in parallel using cilk_for.
    cilk_for (size_t i = 0; i < documents.size(); ++i) {
        const auto& doc = documents[i];
        auto tokens = tokenize_parallel(doc.content);

        // REDUCE Phase: Insert tokens into the appropriate shards.
        // This happens immediately after tokenization for each document.
        for (const auto& token : tokens) {
            size_t shard_index = hash_term_to_shard(token);
            
            // Lock only the specific shard to reduce contention.
            std::lock_guard<std::mutex> lock(shards_[shard_index].mtx);
            shards_[shard_index].postings[token].add_document(doc.id);
        }
    }

    perf_monitor_.end_timer("total_indexing_time");
    std::cout << "Parallel indexing complete.\n";
}

IndexingMetrics ParallelIndexer::get_performance_metrics() const {
    IndexingMetrics metrics;
    double time_ms = perf_monitor_.get_duration_ms("total_indexing_time");
    metrics.indexing_time_ms = time_ms;
    metrics.throughput_docs_per_sec = (time_ms > 0) ? (1000.0 * 50000 / time_ms) : 0;
    return metrics;
}