#ifndef QUERY_H
#define QUERY_H

#include <string>
#include <vector>
#include <memory>

enum class QueryOperator {
    TERM,
    AND,
    OR,
    NOT
};

struct QueryNode {
    QueryOperator op;
    std::string term; // only if op is TERM
    std::vector<std::unique_ptr<QueryNode>> children;

    QueryNode(const std::string &t) : op(QueryOperator::TERM), term(t) {}

    QueryNode(QueryOperator o) : op(o) {}
};
#endif