#ifndef _RETRIEVER_H
#define _RETRIEVER_H

#include "retrieval/result_set.h"
#include "retrieval/query.h"
#include <string>
#include <unordered_map>
#include <vector>
#include <mutex>

struct DiskLocation
{
    long long offset;
    size_t size;
};

struct ShardIndex
{
    std::unordered_map<std::string, DiskLocation> dictionary;
    std::string postings_path;
};

class ParallelRetriever
{
public:
    ParallelRetriever(const std::string &index_path, size_t num_shards);
    ResultSet execute_query(const QueryNode &query);

private:
    ResultSet execute_node(const QueryNode &node, std::unordered_map<std::string, ResultSet> &postings_cache, const ResultSet &context_set);

    std::vector<ShardIndex> shards_;
    std::mutex cache_mutex_;
    ResultSet get_universe(const std::unordered_map<std::string, ResultSet> &postings_cache);
    ResultSet intersect_sets(const ResultSet &a, const ResultSet &b);
    ResultSet union_sets(const ResultSet &a, const ResultSet &b);
    ResultSet differ_sets(const ResultSet &a, const ResultSet &b);
};

#endif 