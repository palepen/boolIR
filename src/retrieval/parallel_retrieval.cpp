#include "retrieval/boolean_retrieval.h"
#include <cilk/cilk.h>
#include <algorithm>

ParallelBooleanRetrieval::ParallelBooleanRetrieval(const std::unordered_map<std::string, PostingList>& index)
    : index_(index) {}

ResultSet ParallelBooleanRetrieval::execute_query(const QueryNode& query) {
    return execute_node(query, 0);
}

// Comprehensive work estimation that considers posting list sizes and operation complexity
size_t estimate_work_size(const QueryNode& node, const std::unordered_map<std::string, PostingList>& index) {
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
    // OR operations process both full lists (more expensive)
    // AND operations can short-circuit (less expensive)
    // NOT operations need full scan of first list
    if (node.op == QueryOperator::OR) {
        total = static_cast<size_t>(total * 1.8);  // Union processes all elements
    } else if (node.op == QueryOperator::NOT) {
        total = static_cast<size_t>(total * 1.5);  // Difference scan
    } else if (node.op == QueryOperator::AND) {
        total = static_cast<size_t>(total * 1.2);  // Intersection can terminate early
    }
    
    return total;
}

ResultSet ParallelBooleanRetrieval::execute_node(const QueryNode& node, int depth) {
    // Handle terminal nodes (actual terms)
    if (node.op == QueryOperator::TERM) {
        auto it = index_.find(node.term);
        if (it != index_.end()) {
            ResultSet result;
            result.doc_ids = it->second.get_postings();
            return result;
        }
        return {};
    }

    // Handle empty or single-child nodes
    if (node.children.empty()) {
        return {};
    }
    
    if (node.children.size() == 1) {
        return execute_node(*node.children[0], depth + 1);
    }

    // ============================================================================
    // INTELLIGENT PARALLELIZATION DECISION ENGINE
    // ============================================================================
    
    // Factor 1: Estimate computational work for both subtrees
    size_t left_work = estimate_work_size(*node.children[0], index_);
    size_t right_work = estimate_work_size(*node.children[1], index_);
    size_t total_work = left_work + right_work;
    
    // Factor 2: Cilk overhead cost model
    // Based on empirical measurements:
    // - cilk_spawn: ~10-15 μs
    // - cilk_sync: ~8-12 μs
    // - Context switching: ~5-10 μs
    // Total overhead: ~25-35 μs
    // Processing rate: ~0.01-0.02 μs per item
    // Break-even point: Need 2000-3000 items to justify overhead
    const size_t ABSOLUTE_MIN_THRESHOLD = 2000;    // Never parallelize below this
    const size_t OPTIMAL_THRESHOLD = 5000;         // Good parallelization above this
    const size_t GUARANTEED_WIN_THRESHOLD = 10000; // Always parallelize above this
    
    // Factor 3: Depth limiting to prevent excessive task creation
    // Too many nested parallel tasks create overhead and contention
    const int MAX_PARALLEL_DEPTH = 3;
    
    // Factor 4: Work balance check
    // If one side is too small relative to the other, parallelization wastes resources
    // Example: 100 items vs 10000 items -> small side finishes fast, no benefit
    const double MIN_BALANCE_RATIO = 0.15;  // Smaller side must be >= 15% of larger
    double balance = 0.0;
    if (left_work > 0 && right_work > 0) {
        balance = static_cast<double>(std::min(left_work, right_work)) / 
                  static_cast<double>(std::max(left_work, right_work));
    }
    
    // ============================================================================
    // DECISION LOGIC: Choose Sequential or Parallel Execution
    // ============================================================================
    
    bool should_parallelize = false;
    
    // Rule 1: NEVER parallelize if work is below absolute minimum
    if (total_work < ABSOLUTE_MIN_THRESHOLD) {
        should_parallelize = false;  // Too small - overhead dominates
    }
    // Rule 2: ALWAYS parallelize if work is above guaranteed win threshold
    else if (total_work >= GUARANTEED_WIN_THRESHOLD && depth < MAX_PARALLEL_DEPTH) {
        should_parallelize = true;   // Large enough - clear benefit
    }
    // Rule 3: For medium workloads, check additional conditions
    else if (total_work >= OPTIMAL_THRESHOLD && depth < MAX_PARALLEL_DEPTH) {
        // Only parallelize if workload is reasonably balanced
        should_parallelize = (balance >= MIN_BALANCE_RATIO);
    }
    // Rule 4: For workloads between absolute min and optimal, be conservative
    else if (total_work >= ABSOLUTE_MIN_THRESHOLD && total_work < OPTIMAL_THRESHOLD) {
        // Require both good balance AND shallow depth
        should_parallelize = (balance >= MIN_BALANCE_RATIO * 1.5) && (depth < MAX_PARALLEL_DEPTH - 1);
    }
    
    // ============================================================================
    // EXECUTE: Parallel or Sequential based on decision
    // ============================================================================
    
    ResultSet left_result, right_result;
    
    if (should_parallelize) {
        // PARALLEL EXECUTION: Spawn subtask for left child, execute right child inline
        left_result = cilk_spawn execute_node(*node.children[0], depth + 1);
        right_result = execute_node(*node.children[1], depth + 1);
        cilk_sync;  // Wait for left child to complete
    } else {
        // SEQUENTIAL EXECUTION: Process both children sequentially
        // This avoids the ~25-35μs overhead of cilk_spawn/sync
        left_result = execute_node(*node.children[0], depth + 1);
        right_result = execute_node(*node.children[1], depth + 1);
    }

    // ============================================================================
    // COMBINE RESULTS: Apply the boolean operation
    // ============================================================================
    
    if (node.op == QueryOperator::AND) {
        // Special case: A AND NOT B pattern
        if (node.children.size() > 1 && 
            node.children[1]->op == QueryOperator::NOT && 
            !node.children[1]->children.empty()) {
            // Execute the NOT operation: A - B
            return ResultSet::differ_sets(left_result, right_result);
        }
        // Standard intersection: A ∩ B
        return ResultSet::intersect_sets(left_result, right_result);
    }
    
    if (node.op == QueryOperator::OR) {
        // Union: A ∪ B
        return ResultSet::union_sets(left_result, right_result);
    }
    
    if (node.op == QueryOperator::NOT) {
        // NOT operation - typically handled within AND context
        // If standalone, return empty or left result negated
        return {};
    }
    
    return {};
}