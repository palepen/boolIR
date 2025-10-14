#include "retrieval/optimized_parallel_retrieval.h"
#include <cilk/cilk.h>
#include <cilk/cilk_stub.h>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <vector>
#include <stdexcept>

OptimizedParallelRetrieval::OptimizedParallelRetrieval(const std::string& dictionary_path, const std::string& postings_path)
    : postings_path_(postings_path) {
    std::cout << "Loading index dictionary from: " << dictionary_path << std::endl;
    load_dictionary(dictionary_path);
}

void OptimizedParallelRetrieval::load_dictionary(const std::string& dictionary_path) {
    std::ifstream dict_file(dictionary_path, std::ios::binary);
    if (!dict_file) {
        throw std::runtime_error("FATAL: Could not open dictionary file at " + dictionary_path);
    }

    std::string term;
    DiskLocation loc;

    while (std::getline(dict_file, term, '\0') &&
           dict_file.read(reinterpret_cast<char*>(&loc.offset), sizeof(loc.offset)) &&
           dict_file.read(reinterpret_cast<char*>(&loc.size), sizeof(loc.size))) {
        dictionary_[term] = loc;
    }
    std::cout << "  -> Dictionary loaded with " << dictionary_.size() << " unique terms." << std::endl;
}

ResultSet OptimizedParallelRetrieval::execute_query_optimized(const QueryNode& query) {
    // Step 1: Traverse the query tree to find all unique terms
    std::vector<const QueryNode*> term_nodes;
    std::vector<const QueryNode*> node_stack;
    node_stack.push_back(&query);
    
    while(!node_stack.empty()){
        const QueryNode* current = node_stack.back();
        node_stack.pop_back();

        if(current->op == QueryOperator::TERM){
            term_nodes.push_back(current);
        }

        for(const auto& child : current->children){
            node_stack.push_back(child.get());
        }
    }

    // Step 2: Fetch posting lists in parallel
    std::unordered_map<std::string, ResultSet> postings_cache;
    
    cilk_for (size_t i = 0; i < term_nodes.size(); ++i) {
        const std::string& term = term_nodes[i]->term;
        auto it = dictionary_.find(term);

        if (it != dictionary_.end()) {
            const DiskLocation& loc = it->second;
            
            ResultSet rs;
            rs.doc_ids.resize(loc.size);

            std::ifstream postings_file(postings_path_, std::ios::binary);
            if (postings_file) {
                postings_file.seekg(loc.offset);
                postings_file.read(reinterpret_cast<char*>(rs.doc_ids.data()), loc.size * sizeof(unsigned int));
            }
            
            { 
                std::lock_guard<std::mutex> lock(file_mutex_);
                postings_cache[term] = std::move(rs);
            }
        }
    }

    // Step 3: Execute boolean logic
    return execute_node_parallel(query, postings_cache);
}

ResultSet OptimizedParallelRetrieval::execute_node_parallel(const QueryNode& node, 
                                                            std::unordered_map<std::string, ResultSet>& postings_cache) {
    if (node.op == QueryOperator::TERM) {
        auto it = postings_cache.find(node.term);
        return (it != postings_cache.end()) ? it->second : ResultSet{};
    }

    if (node.children.empty()) {
        return {};
    }

    std::vector<ResultSet> child_results(node.children.size());

    cilk_for (size_t i = 0; i < node.children.size(); ++i) {
        child_results[i] = execute_node_parallel(*node.children[i], postings_cache);
    }
    
    ResultSet result = child_results[0];
    for (size_t i = 1; i < child_results.size(); ++i) {
        if (node.op == QueryOperator::AND) {
            if (result.doc_ids.size() > child_results[i].doc_ids.size()) {
                 result = intersect_sets(child_results[i], result);
            } else {
                 result = intersect_sets(result, child_results[i]);
            }
        } else if (node.op == QueryOperator::OR) {
            result = union_sets(result, child_results[i]);
        }
    }
    
    return result;
}

ResultSet OptimizedParallelRetrieval::intersect_sets(const ResultSet& a, const ResultSet& b) {
    ResultSet result;
    result.doc_ids.reserve(std::min(a.doc_ids.size(), b.doc_ids.size()));
    std::set_intersection(a.doc_ids.begin(), a.doc_ids.end(),
                          b.doc_ids.begin(), b.doc_ids.end(),
                          std::back_inserter(result.doc_ids));
    return result;
}

ResultSet OptimizedParallelRetrieval::union_sets(const ResultSet& a, const ResultSet& b) {
    ResultSet result;
    result.doc_ids.reserve(a.doc_ids.size() + b.doc_ids.size());
    std::set_union(a.doc_ids.begin(), a.doc_ids.end(),
                   b.doc_ids.begin(), b.doc_ids.end(),
                   std::back_inserter(result.doc_ids));
    return result;
}

ResultSet OptimizedParallelRetrieval::differ_sets(const ResultSet& a, const ResultSet& b) {
    ResultSet result;
    result.doc_ids.reserve(a.doc_ids.size());
    std::set_difference(a.doc_ids.begin(), a.doc_ids.end(),
                        b.doc_ids.begin(), b.doc_ids.end(),
                        std::back_inserter(result.doc_ids));
    return result;
}