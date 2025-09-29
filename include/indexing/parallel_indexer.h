#ifndef PARALLEL_INDEXER_H
#define PARALLEL_INDEXER_H

#include <string>
#include <vector>
#include <unordered_map>
#include <mutex>

#include "document.h"
#include "posting_list.h"
#include "performance_monitor.h"


class ParallelIndexer {
    public:
        explicit ParallelIndexer(size_t num_shards = 16);
        void build_index_parallel(const DocumentCollection &documents);
        
        IndexingMetrics get_performance_metrics() const;
    private:

        /*
            A shard is a self-contained, thread safe partition of the main index
        */
        struct Shard{
            std::unordered_map<std::string, PostingList> postings;
            mutable std::mutex mtx;
        };

        // hashes a term tp determine which shard it belongs to
        size_t hash_term_to_shard(const std::string& term) const;

        std::vector<Shard> shards_;
        mutable PerformanceMonitor perf_monitor_;

    };
#endif
