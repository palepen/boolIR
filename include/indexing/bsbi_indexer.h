#ifndef BSBI_INDEXER_H
#define BSBI_INDEXER_H

#include "indexing/document.h"
#include "indexing/performance_monitor.h"
#include "data_loader.h"  // For IdToDocNameMap
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
                const IdToDocNameMap& id_to_doc_name,  
                const std::string& index_path,
                const std::string& temp_path,
                size_t block_size_mb = 256,
                size_t num_shards = 64, 
                size_t num_workers = 0);

    void build_index();
    
private:
    std::vector<std::string> generate_runs();
    std::string merge_runs(std::vector<std::string>& run_files);
    void create_sharded_index_files(const std::string& final_run_path); 
    void create_document_store();
    void print_indexing_summary();
    size_t get_effective_workers() const;

    const DocumentCollection& documents_;
    const IdToDocNameMap& id_to_doc_name_;  
    std::string index_path_;
    std::string temp_path_;
    size_t block_size_bytes_;
    size_t num_shards_; 
    size_t num_workers_;
    PerformanceMonitor perf_monitor_;
    std::mutex vector_mutex_;
};

#endif