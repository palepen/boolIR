#ifndef RETRIEVAL_H
#define RETRIEVAL_H

#include "indexing.h"
#include <string>
#include <vector>

// --- Query Representations ---

// (For Sequential Search)
typedef struct BooleanQuery {
    char** terms;
    char* operators;
    int num_terms;
} BooleanQuery;

// (For Parallel Search)
typedef struct QueryNode {
    char* term;
    char op;
    QueryNode* left;
    QueryNode* right;
} QueryNode;

// --- Result Structures (shared) ---

typedef struct SearchResult {
    uint32_t doc_id;
    double score;
    char* snippet;
} SearchResult;

typedef struct ResultSet {
    SearchResult* results;
    int num_results;
    int capacity;
} ResultSet;


// --- Function Declarations ---

// Sequential (Baseline) Search
BooleanQuery* parse_boolean_query(const char* query_string);
void free_boolean_query(BooleanQuery* query);
ResultSet* execute_sequential_search(InvertedIndex* index, BooleanQuery* query);

// Parallel Search
QueryNode* parse_query_to_tree(const std::string& query_string);
void free_query_tree(QueryNode* node);
ResultSet* execute_parallel_search(InvertedIndex* index, QueryNode* root);

// Utility
void free_result_set(ResultSet* results);

#endif // RETRIEVAL_H