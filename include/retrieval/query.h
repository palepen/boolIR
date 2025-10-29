#ifndef QUERY_H
#define QUERY_H

#include <string>
#include <vector>
#include <memory>
#include <sstream>

enum class QueryOperator {
    TERM,
    AND,
    OR,
    NOT
};

struct QueryNode {
    QueryOperator op;
    std::string term;
    std::vector<std::unique_ptr<QueryNode>> children;

    QueryNode(const std::string &t) : op(QueryOperator::TERM), term(t) {}

    QueryNode(QueryOperator o) : op(o) {}

    void to_string(std::stringstream& ss, int indent = 0) const {
        ss << std::string(indent * 2, ' '); // Add indentation

        switch (op) {
            case QueryOperator::TERM:
                ss << "TERM(\"" << term << "\")\n";
                break;
            case QueryOperator::AND:
                ss << "AND\n";
                break;
            case QueryOperator::OR:
                ss << "OR\n";
                break;
            case QueryOperator::NOT:
                ss << "NOT\n";
                break;
        }

        // Recursively print children
        for (const auto& child : children) {
            child->to_string(ss, indent + 1);
        }
    }
};
#endif