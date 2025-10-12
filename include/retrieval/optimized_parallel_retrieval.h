#ifndef OPTIMIZED_PARALLEL_RETRIEVAL_H
#define OPTIMIZED_PARALLEL_RETRIEVAL_H

#include "retrieval/result_set.h"
#include "retrieval/query.h"
#include <string>
#include <unordered_map>
#include <vector>
#include <mutex>

struct DiskLocation {
    long long offset;
    size_t size;
};

/**
 * Pure Boolean Retrieval Engine
 * Returns documents that match Boolean query criteria
 * without scoring or ranking
 */
class OptimizedParallelRetrieval {
public:
    OptimizedParallelRetrieval(const std::string& dictionary_path, const std::string& postings_path);

    ResultSet execute_query_optimized(const QueryNode& query);

private:
    ResultSet execute_node_parallel(const QueryNode& node, 
                                    std::unordered_map<std::string, ResultSet>& postings_cache);
    
    void load_dictionary(const std::string& dictionary_path);

    std::unordered_map<std::string, DiskLocation> dictionary_;
    std::string postings_path_;
    std::mutex file_mutex_;

    // Pure Boolean set operations
    ResultSet intersect_sets(const ResultSet& a, const ResultSet& b);
    ResultSet union_sets(const ResultSet& a, const ResultSet& b);
    ResultSet differ_sets(const ResultSet& a, const ResultSet& b);
};

#endif