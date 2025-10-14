#ifndef BSBI_INDEXER_H
#define BSBI_INDEXER_H

#include "indexing/document.h"
#include "indexing/performance_monitor.h"
#include <string>
#include <vector>
#include <mutex>

struct TermDocPair {
    std::string term;
    unsigned int doc_id;

    bool operator<(const TermDocPair& other) const {
        if (term != other.term) {
            return term < other.term;
        }
        return doc_id < other.doc_id;
    }
};

/**
 * BSBI Indexer with Performance Monitoring
 * Pure Boolean inverted index construction with detailed metrics
 */
class BSBIIndexer {
public:
    BSBIIndexer(const DocumentCollection& documents,
                const std::string& index_path,
                const std::string& temp_path,
                size_t block_size_mb = 256);

    void build_index();

private:
    std::vector<std::string> generate_runs();
    std::string merge_runs(std::vector<std::string>& run_files);
    void create_final_index_files(const std::string& final_run_path);
    void print_indexing_summary();

    const DocumentCollection& documents_;
    std::string index_path_;
    std::string temp_path_;
    size_t block_size_bytes_;
    PerformanceMonitor perf_monitor_;
    std::mutex vector_mutex_;
};

#endif