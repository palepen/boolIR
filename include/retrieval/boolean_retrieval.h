#ifndef BOOLEAN_RETRIEVAL_H
#define BOOLEAN_RETRIEVAL_H

#include <unordered_map>

#include "query.h"
#include "result_set.h"
#include "indexing/posting_list.h"

class BooleanRetrieval
{
public:
    virtual ~BooleanRetrieval() = default;
    virtual ResultSet execute_query(const QueryNode &query) = 0;
};

class SequentialBooleanRetrieval : public BooleanRetrieval
{
public:
    explicit SequentialBooleanRetrieval(const std::unordered_map<std::string, PostingList> &index);
    ResultSet execute_query(const QueryNode &query) override;

private:
    ResultSet execute_node(const QueryNode &node);
    const std::unordered_map<std::string, PostingList> &index_;
};

class ParallelBooleanRetrieval : public BooleanRetrieval
{
public:
    explicit ParallelBooleanRetrieval(const std::unordered_map<std::string, PostingList> &index);
    ResultSet execute_query(const QueryNode &query) override;

private:
    ResultSet execute_node(const QueryNode& node, int depth);
    const std::unordered_map<std::string, PostingList>& index_;
};

#endif