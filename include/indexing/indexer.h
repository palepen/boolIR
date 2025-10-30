#ifndef INDEXER_H
#define INDEXER_H

#include "indexing/document_stream.h"
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
 * Streaming BSBI Indexer - Memory-Efficient Version
 * 
 * Key improvements over original:
 * - Documents are streamed from disk, not loaded into RAM
 * - Uses memory-mapped files for efficient I/O
 * - Memory usage = num_workers × block_size_bytes (constant, independent of corpus size)
 * - Can index arbitrarily large corpora that exceed RAM
 * 
 * Parallelization strategy:
 * - Document-level partitioning: each worker processes a disjoint range of doc IDs
 * - Workers stream documents independently using memory mapping
 * - No shared state during run generation (embarrassingly parallel)
 * - Merge phase uses parallel k-way merge (already implemented)
 */
class Indexer {
public:
    /**
     * @param doc_stream Document stream providing on-demand access to corpus
     * @param index_path Output directory for index files
     * @param temp_path Temporary directory for intermediate run files
     * @param block_size_mb Maximum memory per worker (in MB)
     * @param num_shards Number of index shards to create
     * @param num_workers Number of parallel workers (0 = auto-detect)
     */
    Indexer(
        const DocumentStream& doc_stream,
        const std::string& index_path,
        const std::string& temp_path,
        size_t block_size_mb = 256,
        size_t num_shards = 64,
        size_t num_workers = 0
    );

    /**
     * Build the complete index using streaming approach
     */
    void build_index();
    
private:
    /**
     * Phase 1: Generate sorted runs by streaming documents
     * Each worker processes a disjoint range of documents
     */
    std::vector<std::string> generate_runs_streaming();
    
    /**
     * Phase 2: Merge runs (unchanged from original)
     */
    std::string merge_runs(std::vector<std::string>& run_files);
    
    /**
     * Phase 3: Create sharded index files (unchanged from original)
     */
    void create_sharded_index_files(const std::string& final_run_path);
    
    /**
     * Phase 4: Create document store (unchanged from original)
     */
    void create_document_store();
    
    /**
     * Print performance summary
     */
    void print_indexing_summary();
    
    /**
     * Get effective number of workers (auto-detect if num_workers_ = 0)
     */
    size_t get_effective_workers() const;

    const DocumentStream& doc_stream_;
    std::string index_path_;
    std::string temp_path_;
    size_t block_size_bytes_;
    size_t num_shards_;
    size_t num_workers_;
    PerformanceMonitor perf_monitor_;
    std::mutex vector_mutex_;  // For thread-safe run file list updates
};

#endif // INDEXER_H