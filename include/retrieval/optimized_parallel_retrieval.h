#ifndef OPTIMIZED_PARALLEL_RETRIEVAL_H
#define OPTIMIZED_PARALLEL_RETRIEVAL_H

#include "retrieval/result_set.h"
#include "indexing/posting_list.h"
#include "retrieval/query.h"
#include <unordered_map>
#include <vector>
#include <immintrin.h>
#include <cstdint>

class ParallelRetrieval {
private:
    const std::unordered_map<std::string, PostingList>& index_;
    struct AlignedPostingList {
        alignas(64) std::vector<uint32_t> doc_ids;
    };
    std::unordered_map<std::string, AlignedPostingList> aligned_index_;

    ResultSet execute_node_parallel(const QueryNode& node, int depth);
    ResultSet combine_results_optimized(QueryOperator op, const ResultSet& left, const ResultSet& right);
    ResultSet intersect_optimized(const ResultSet& a, const ResultSet& b);
    ResultSet union_optimized(const ResultSet& a, const ResultSet& b);
    ResultSet difference_optimized(const ResultSet& a, const ResultSet& b);
    ResultSet combine_multi_results(QueryOperator op, const std::vector<ResultSet>& results);
    ResultSet parallel_union_reduce(const std::vector<ResultSet>& results);
    ResultSet merge_sorted_results(const ResultSet& a, const ResultSet& b);
    std::vector<uint32_t> simd_intersect(const std::vector<uint32_t>& a, const std::vector<uint32_t>& b);
    size_t estimate_work_size(const QueryNode& node, const std::unordered_map<std::string, PostingList>& index);

public:
    ParallelRetrieval(const std::unordered_map<std::string, PostingList>& index);
    ResultSet execute_query_optimized(const QueryNode& query);
};

#endif