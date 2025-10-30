#include "retrieval/retriever.h"
#include <cilk/cilk.h>
#include <cilk/cilk_stub.h>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <stdexcept>
#include <filesystem>

namespace fs = std::filesystem;

// A struct to represent one unit of work: fetching one term from one shard
struct RetrievalTask
{
    std::string term;
    size_t shard_id;
};

ParallelRetriever::ParallelRetriever(const std::string &index_path, size_t num_shards)
{
    shards_.resize(num_shards);
    std::cout << "Loading " << num_shards << " index shards..." << std::endl;

    for (size_t s = 0; s < num_shards; ++s)
    {
        std::string shard_dir = index_path + "/shard_" + std::to_string(s);
        std::string dict_path = shard_dir + "/dict.dat";

        if (!fs::exists(dict_path))
        {
            throw std::runtime_error("Shard dictionary not found: " + dict_path);
        }

        shards_[s].postings_path = shard_dir + "/postings.dat";

        std::ifstream dict_file(dict_path, std::ios::binary);
        std::string term;
        DiskLocation loc;

        while (std::getline(dict_file, term, '\0') &&
               dict_file.read(reinterpret_cast<char *>(&loc.offset), sizeof(loc.offset)) &&
               dict_file.read(reinterpret_cast<char *>(&loc.size), sizeof(loc.size)))
        {
            shards_[s].dictionary[term] = loc;
        }
    }
    std::cout << "All shards loaded." << std::endl;
}

ResultSet ParallelRetriever::execute_query(const QueryNode &query)
{
    // Traverse the query tree to find all unique terms needed
    std::vector<std::string> unique_terms;
    std::vector<const QueryNode *> node_stack;
    node_stack.push_back(&query);

    while (!node_stack.empty())
    {
        const QueryNode *current = node_stack.back();
        node_stack.pop_back();
        if (current->op == QueryOperator::TERM)
            unique_terms.push_back(current->term);
        for (const auto &child : current->children)
            node_stack.push_back(child.get());
    }
    std::sort(unique_terms.begin(), unique_terms.end());
    unique_terms.erase(std::unique(unique_terms.begin(), unique_terms.end()), unique_terms.end());

    // Create a list of all retrieval tasks
    std::vector<RetrievalTask> tasks;
    for (const auto &term : unique_terms)
    {
        size_t shard_id = std::hash<std::string>{}(term) % shards_.size();
        if (shards_[shard_id].dictionary.count(term))
        {
            tasks.push_back({term, shard_id});
        }
    }

    // Fetch all posting lists in parallel using the task list
    std::unordered_map<std::string, ResultSet> postings_cache;
    cilk_for(size_t i = 0; i < tasks.size(); ++i)
    {
        const auto &task = tasks[i];
        const DiskLocation &loc = shards_[task.shard_id].dictionary.at(task.term);

        ResultSet rs;
        rs.doc_ids.resize(loc.size);

        std::ifstream postings_file(shards_[task.shard_id].postings_path, std::ios::binary);
        if (postings_file)
        {
            postings_file.seekg(loc.offset);
            postings_file.read(reinterpret_cast<char *>(rs.doc_ids.data()), loc.size * sizeof(unsigned int));
        }

        {
            std::lock_guard<std::mutex> lock(cache_mutex_);
            postings_cache[task.term] = std::move(rs);
        }
    }

    return execute_node(query, postings_cache, ResultSet{});
}

ResultSet ParallelRetriever::execute_node(const QueryNode &node,
                                          std::unordered_map<std::string, ResultSet> &postings_cache,
                                          const ResultSet &context_set)
{

    if (node.op == QueryOperator::TERM)
    {
        auto it = postings_cache.find(node.term);
        return (it != postings_cache.end()) ? it->second : ResultSet{};
    }

    if (node.children.empty())
        return {};

    if (node.op == QueryOperator::NOT)
    {
        if (node.children.size() != 1)
        {
            std::cerr << "Warning: NOT operator requires exactly one child" << std::endl;
            return {};
        }

        ResultSet child_result = execute_node(*node.children[0], postings_cache, context_set);

        if (context_set.doc_ids.empty())
        {
            ResultSet universe = get_universe(postings_cache);
            return differ_sets(universe, child_result);
        }
        else
        {
            return differ_sets(context_set, child_result);
        }
    }

    if (node.op == QueryOperator::AND)
    {
        ResultSet result;

        for (size_t i = 0; i < node.children.size(); ++i)
        {
            const auto &child = node.children[i];

            if (i == 0 || result.doc_ids.empty())
            {
                result = execute_node(*child, postings_cache, ResultSet{});
            }
            else if (child->op == QueryOperator::NOT)
            {
                ResultSet not_result = execute_node(*child, postings_cache, result);
                result = not_result;
            }
            else
            {
                ResultSet child_result = execute_node(*child, postings_cache, ResultSet{});
                result = intersect_sets(result, child_result);
            }
        }

        return result;
    }

    if (node.op == QueryOperator::OR)
    {
        std::vector<ResultSet> child_results;
        child_results.reserve(node.children.size());

        for (const auto &child : node.children)
        {
            child_results.push_back(execute_node(*child, postings_cache, ResultSet{}));
        }

        ResultSet result = child_results[0];
        for (size_t i = 1; i < child_results.size(); ++i)
        {
            result = union_sets(result, child_results[i]);
        }
        return result;
    }

    return {};
}

ResultSet ParallelRetriever::get_universe(const std::unordered_map<std::string, ResultSet> &postings_cache)
{
    ResultSet universe;

    if (postings_cache.empty())
        return universe;

    auto it = postings_cache.begin();
    universe = it->second;
    ++it;

    for (; it != postings_cache.end(); ++it)
    {
        universe = union_sets(universe, it->second);
    }

    return universe;
}

ResultSet ParallelRetriever::intersect_sets(const ResultSet &a, const ResultSet &b)
{
    return ResultSet::intersect_sets(a, b);
}

ResultSet ParallelRetriever::union_sets(const ResultSet &a, const ResultSet &b)
{
    return ResultSet::union_sets(a, b);
}

ResultSet ParallelRetriever::differ_sets(const ResultSet &a, const ResultSet &b)
{
    return ResultSet::differ_sets(a, b);
}