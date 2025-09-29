#include "indexing/sequential_indexer.h"
#include <sstream>
#include <iostream>

// Simple tokenizer: splits by space and converts to lowercase.
static std::vector<std::string> tokenize(const std::string& text) {
    std::vector<std::string> tokens;
    std::stringstream ss(text);
    std::string token;
    while (ss >> token) {
        // Simple case conversion
        for (char &c : token) {
            c = tolower(c);
        }
        tokens.push_back(token);
    }
    return tokens;
}

void SequentialIndexer::build_index(const DocumentCollection& documents) {
    std::cout << "Starting sequential indexing for " << documents.size() << " documents...\n";
    perf_monitor_.start_timer("total_indexing_time");

    for (const auto& doc : documents) {
        auto tokens = tokenize(doc.content);
        for (const auto& token : tokens) {
            inverted_index_[token].add_document(doc.id);
        }
    }

    perf_monitor_.end_timer("total_indexing_time");
    std::cout << "Sequential indexing complete.\n";
}

IndexingMetrics SequentialIndexer::get_performance_metrics() const {
    IndexingMetrics metrics;
    double time_ms = perf_monitor_.get_duration_ms("total_indexing_time");
    metrics.indexing_time_ms = time_ms;
    metrics.throughput_docs_per_sec = (time_ms > 0) ? (1000.0 * 50000 / time_ms) : 0;
    return metrics;
}