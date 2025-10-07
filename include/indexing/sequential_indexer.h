#ifndef SEQUENTIAL_INDEXER_H
#define SEQUENTIAL_INDEXER_H

#include <string>
#include <vector>
#include <unordered_map>

#include "document.h"
#include "posting_list.h"
#include "performance_monitor.h"

class SequentialIndexer {
public:
    void build_index(const DocumentCollection& documents);

    IndexingMetrics get_performance_metrics() const;

private:
    std::unordered_map<std::string, PostingList> inverted_index_;
    mutable PerformanceMonitor perf_monitor_;
    size_t num_docs_indexed_ = 0;

};

#endif