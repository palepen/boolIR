#include "retrieval/optimized_parallel_retrieval.h"
#include <cilk/cilk.h>
#include <algorithm>
#include <unordered_set>

ParallelRetrieval::ParallelRetrieval(const std::unordered_map<std::string, PostingList>& index) 
    : index_(index) {
    // Pre-process index for better cache performance
    for (const auto& [term, posting_list] : index) {
        AlignedPostingList aligned;
        aligned.doc_ids = posting_list.get_postings();
        aligned_index_[term] = std::move(aligned);
    }
}

size_t ParallelRetrieval::estimate_work_size(const QueryNode& node, const std::unordered_map<std::string, PostingList>& index) {
    if (node.op == QueryOperator::TERM) {
        auto it = index.find(node.term);
        return (it != index.end()) ? it->second.get_postings().size() : 0;
    }
    
    if (node.children.empty()) {
        return 0;
    }
    
    size_t total = 0;
    for (const auto& child : node.children) {
        total += estimate_work_size(*child, index);
    }
    
    // Weight operations by their computational complexity
    if (node.op == QueryOperator::OR) {
        total = static_cast<size_t>(total * 1.8);
    } else if (node.op == QueryOperator::NOT) {
        total = static_cast<size_t>(total * 1.5);
    } else if (node.op == QueryOperator::AND) {
        total = static_cast<size_t>(total * 1.2);
    }
    
    return total;
}

std::vector<uint32_t> ParallelRetrieval::simd_intersect(
    const std::vector<uint32_t>& a, 
    const std::vector<uint32_t>& b
) {
    std::vector<uint32_t> result;
    result.reserve(std::min(a.size(), b.size()));
    
    size_t i = 0, j = 0;
    
    // Use galloping search for very different sizes
    if (a.size() > b.size() * 10) {
        for (uint32_t val : b) {
            auto it = std::lower_bound(a.begin() + i, a.end(), val);
            if (it != a.end() && *it == val) {
                result.push_back(val);
                i = it - a.begin() + 1;
            }
        }
        return result;
    }
    
    // Standard merge with prefetching
    while (i < a.size() && j < b.size()) {
        // Prefetch next cache lines
        if (i + 8 < a.size()) __builtin_prefetch(&a[i + 8], 0, 1);
        if (j + 8 < b.size()) __builtin_prefetch(&b[j + 8], 0, 1);
        
        if (a[i] < b[j]) {
            i++;
        } else if (b[j] < a[i]) {
            j++;
        } else {
            result.push_back(a[i]);
            i++; j++;
        }
    }
    
    return result;
}

ResultSet ParallelRetrieval::execute_query_optimized(const QueryNode& query) {
    return execute_node_parallel(query, 0);
}

ResultSet ParallelRetrieval::execute_node_parallel(const QueryNode& node, int depth) {
    // Terminal node - fetch posting list
    if (node.op == QueryOperator::TERM) {
        auto it = aligned_index_.find(node.term);
        if (it != aligned_index_.end()) {
            ResultSet result;
            result.doc_ids = it->second.doc_ids;
            return result;
        }
        return {};
    }
    
    if (node.children.empty()) return {};
    if (node.children.size() == 1) {
        return execute_node_parallel(*node.children[0], depth + 1);
    }
    
    // PARALLEL QUERY TERM FETCHING
    // For multi-term queries, fetch posting lists in parallel
    if (node.children.size() >= 3 && depth == 0) {
        // Parallel term fetching for multi-word queries
        std::vector<ResultSet> term_results(node.children.size());
        
        cilk_for(size_t i = 0; i < node.children.size(); ++i) {
            term_results[i] = execute_node_parallel(*node.children[i], depth + 1);
        }
        
        // Merge results based on operator
        return combine_multi_results(node.op, term_results);
    }
    
    // Adaptive parallelization based on work size
    size_t total_work = estimate_work_size(node, index_);
    
    // Different thresholds for different operators
    size_t parallel_threshold = 5000;
    if (node.op == QueryOperator::OR) {
        parallel_threshold = 3000;  // OR benefits more from parallelism
    } else if (node.op == QueryOperator::AND) {
        parallel_threshold = 7000;  // AND can short-circuit
    }
    
    if (total_work >= parallel_threshold && depth < 4) {
        // Parallel execution for large workloads
        if (node.children.size() == 2) {
            ResultSet left_result, right_result;
            
            left_result = cilk_spawn execute_node_parallel(*node.children[0], depth + 1);
            right_result = execute_node_parallel(*node.children[1], depth + 1);
            cilk_sync;
            
            return combine_results_optimized(node.op, left_result, right_result);
        } else {
            // Multiple children - parallel processing
            std::vector<ResultSet> child_results(node.children.size());
            
            cilk_for(size_t i = 0; i < node.children.size(); ++i) {
                child_results[i] = execute_node_parallel(*node.children[i], depth + 1);
            }
            
            return combine_multi_results(node.op, child_results);
        }
    } else {
        // Sequential execution for small workloads
        ResultSet result = execute_node_parallel(*node.children[0], depth + 1);
        
        for (size_t i = 1; i < node.children.size(); ++i) {
            ResultSet child_result = execute_node_parallel(*node.children[i], depth + 1);
            result = combine_results_optimized(node.op, result, child_result);
            
            // Early termination for AND with empty result
            if (node.op == QueryOperator::AND && result.doc_ids.empty()) {
                break;
            }
        }
        
        return result;
    }
}

ResultSet ParallelRetrieval::combine_results_optimized(
    QueryOperator op,
    const ResultSet& left,
    const ResultSet& right
) {
    switch (op) {
        case QueryOperator::AND:
            return intersect_optimized(left, right);
        case QueryOperator::OR:
            return union_optimized(left, right);
        case QueryOperator::NOT:
            return difference_optimized(left, right);
        default:
            return {};
    }
}

ResultSet ParallelRetrieval::intersect_optimized(const ResultSet& a, const ResultSet& b) {
    ResultSet result;
    result.doc_ids = simd_intersect(a.doc_ids, b.doc_ids);
    return result;
}

ResultSet ParallelRetrieval::union_optimized(const ResultSet& a, const ResultSet& b) {
    if (a.doc_ids.empty()) return b;
    if (b.doc_ids.empty()) return a;
    
    ResultSet result;
    result.doc_ids.reserve(a.doc_ids.size() + b.doc_ids.size());
    
    // Parallel merge for large lists
    if (a.doc_ids.size() + b.doc_ids.size() > 10000) {
        size_t mid_a = a.doc_ids.size() / 2;
        size_t mid_b = b.doc_ids.size() / 2;
        
        ResultSet left_part = cilk_spawn union_optimized(
            ResultSet{std::vector<uint32_t>(a.doc_ids.begin(), a.doc_ids.begin() + mid_a)},
            ResultSet{std::vector<uint32_t>(b.doc_ids.begin(), b.doc_ids.begin() + mid_b)}
        );
        
        ResultSet right_part = union_optimized(
            ResultSet{std::vector<uint32_t>(a.doc_ids.begin() + mid_a, a.doc_ids.end())},
            ResultSet{std::vector<uint32_t>(b.doc_ids.begin() + mid_b, b.doc_ids.end())}
        );
        
        cilk_sync;
        
        return merge_sorted_results(left_part, right_part);
    } else {
        // Sequential merge for small lists
        std::merge(a.doc_ids.begin(), a.doc_ids.end(),
                   b.doc_ids.begin(), b.doc_ids.end(),
                   std::back_inserter(result.doc_ids));
        
        // Remove duplicates
        result.doc_ids.erase(
            std::unique(result.doc_ids.begin(), result.doc_ids.end()),
            result.doc_ids.end()
        );
    }
    
    return result;
}

ResultSet ParallelRetrieval::difference_optimized(const ResultSet& a, const ResultSet& b) {
    if (a.doc_ids.empty() || b.doc_ids.empty()) return a;
    
    ResultSet result;
    result.doc_ids.reserve(a.doc_ids.size());
    
    // Use hash set for O(1) lookups if b is large
    if (b.doc_ids.size() > 1000) {
        std::unordered_set<uint32_t> b_set(b.doc_ids.begin(), b.doc_ids.end());
        for (uint32_t id : a.doc_ids) {
            if (b_set.find(id) == b_set.end()) {
                result.doc_ids.push_back(id);
            }
        }
    } else {
        // Standard set difference for small lists
        std::set_difference(a.doc_ids.begin(), a.doc_ids.end(),
                            b.doc_ids.begin(), b.doc_ids.end(),
                            std::back_inserter(result.doc_ids));
    }
    
    return result;
}

ResultSet ParallelRetrieval::combine_multi_results(
    QueryOperator op,
    const std::vector<ResultSet>& results
) {
    if (results.empty()) return {};
    if (results.size() == 1) return results[0];
    
    if (op == QueryOperator::AND) {
        // For AND, start with smallest list for early termination
        size_t min_idx = 0;
        size_t min_size = results[0].doc_ids.size();
        for (size_t i = 1; i < results.size(); ++i) {
            if (results[i].doc_ids.size() < min_size) {
                min_size = results[i].doc_ids.size();
                min_idx = i;
            }
        }
        
        ResultSet result = results[min_idx];
        for (size_t i = 0; i < results.size(); ++i) {
            if (i != min_idx) {
                result = intersect_optimized(result, results[i]);
                if (result.doc_ids.empty()) break;
            }
        }
        return result;
    } else if (op == QueryOperator::OR) {
        // Parallel reduction for OR
        return parallel_union_reduce(results);
    }
    
    return {};
}

ResultSet ParallelRetrieval::parallel_union_reduce(const std::vector<ResultSet>& results) {
    if (results.size() == 1) return results[0];
    if (results.size() == 2) return union_optimized(results[0], results[1]);
    
    // Divide and conquer
    size_t mid = results.size() / 2;
    
    ResultSet left = cilk_spawn parallel_union_reduce(
        std::vector<ResultSet>(results.begin(), results.begin() + mid)
    );
    
    ResultSet right = parallel_union_reduce(
        std::vector<ResultSet>(results.begin() + mid, results.end())
    );
    
    cilk_sync;
    
    return union_optimized(left, right);
}

ResultSet ParallelRetrieval::merge_sorted_results(const ResultSet& a, const ResultSet& b) {
    ResultSet result;
    std::merge(a.doc_ids.begin(), a.doc_ids.end(),
               b.doc_ids.begin(), b.doc_ids.end(),
               std::back_inserter(result.doc_ids));
    return result;
}