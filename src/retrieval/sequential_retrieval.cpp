#include "retrieval/boolean_retrieval.h"

SequentialBooleanRetrieval::SequentialBooleanRetrieval(const std::unordered_map<std::string, PostingList>& index)
    : index_(index) {}

ResultSet SequentialBooleanRetrieval::execute_query(const QueryNode& query) {
    return execute_node(query);
}

ResultSet SequentialBooleanRetrieval::execute_node(const QueryNode& node) {
    if (node.op == QueryOperator::TERM) {
        auto it = index_.find(node.term);
        if (it != index_.end()) {
            ResultSet result;
            result.doc_ids = it->second.get_postings();
            return result;
        }
        return {};
    }

    if (node.op == QueryOperator::OR) {
        ResultSet result;
        for (const auto& child : node.children) {
            result = ResultSet::union_sets(result, execute_node(*child));
        }
        return result;
    }
    
    if (node.op == QueryOperator::AND) {
        if (node.children.empty()) return {};
        ResultSet result = execute_node(*node.children[0]);
        for (size_t i = 1; i < node.children.size(); ++i) {
            const auto& child_node = *node.children[i];
            if (child_node.op == QueryOperator::NOT && !child_node.children.empty()) {
                ResultSet to_subtract = execute_node(*child_node.children[0]);
                result = ResultSet::differ_sets(result, to_subtract);
            } else {
                result = ResultSet::intersect_sets(result, execute_node(child_node));
            }
        }
        return result;
    }
    
    return {};
}