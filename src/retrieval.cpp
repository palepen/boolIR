#include <iostream>
#include <string>
#include <cstring>
#include <vector>
#include <sstream>
#include <algorithm>
#include <stack>
#include <cilk/cilk.h>
#include "retrieval.h"

// --- Helper Functions & Posting List Ops ---

static std::vector<std::string> split(const std::string& s, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);
    while (std::getline(tokenStream, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

static PostingList* intersect_postings(PostingList* p1, PostingList* p2) {
    PostingList dummy_head = {0, 0, nullptr};
    PostingList* last = &dummy_head;
    while (p1 != nullptr && p2 != nullptr) {
        if (p1->doc_id == p2->doc_id) {
            last->next = new PostingList{p1->doc_id, 0, nullptr}; last = last->next;
            p1 = p1->next; p2 = p2->next;
        } else if (p1->doc_id < p2->doc_id) { p1 = p1->next; }
        else { p2 = p2->next; }
    }
    return dummy_head.next;
}

static PostingList* union_postings(PostingList* p1, PostingList* p2) {
    PostingList dummy_head = {0, 0, nullptr};
    PostingList* last = &dummy_head;
    while (p1 != nullptr || p2 != nullptr) {
        if (p1 != nullptr && (p2 == nullptr || p1->doc_id < p2->doc_id)) {
            last->next = new PostingList{p1->doc_id, 0, nullptr}; p1 = p1->next;
        } else if (p2 != nullptr && (p1 == nullptr || p2->doc_id < p1->doc_id)) {
            last->next = new PostingList{p2->doc_id, 0, nullptr}; p2 = p2->next;
        } else {
            last->next = new PostingList{p1->doc_id, 0, nullptr};
            p1 = p1->next; p2 = p2->next;
        }
        last = last->next;
    }
    return dummy_head.next;
}


// =================================================================
//  SECTION 1: SEQUENTIAL (BASELINE) SEARCH IMPLEMENTATION
// =================================================================

BooleanQuery* parse_boolean_query(const char* query_string) {
    BooleanQuery* query = new BooleanQuery;
    auto parts = split(std::string(query_string), ' ');
    std::vector<char*> terms;
    std::string operators_str;
    for (const auto& part : parts) {
        if (part == "AND" || part == "OR") {
            operators_str += part[0];
        } else {
            char* term = new char[part.length() + 1];
            strcpy(term, part.c_str());
            terms.push_back(term);
        }
    }
    query->num_terms = terms.size();
    query->terms = new char*[query->num_terms];
    std::copy(terms.begin(), terms.end(), query->terms);
    query->operators = new char[operators_str.length() + 1];
    strcpy(query->operators, operators_str.c_str());
    return query;
}

void free_boolean_query(BooleanQuery* query) {
    if (!query) return;
    for (int i = 0; i < query->num_terms; ++i) delete[] query->terms[i];
    delete[] query->terms;
    delete[] query->operators;
    delete query;
}

ResultSet* execute_sequential_search(InvertedIndex* index, BooleanQuery* query) {
    if (!query || query->num_terms == 0) return nullptr;
    PostingList* result_list = get_posting_list(index, query->terms[0]);
    for (int i = 0; i < query->num_terms - 1; ++i) {
        PostingList* next_list = get_posting_list(index, query->terms[i + 1]);
        if (query->operators[i] == 'A') result_list = intersect_postings(result_list, next_list);
        else if (query->operators[i] == 'O') result_list = union_postings(result_list, next_list);
    }
    ResultSet* results = new ResultSet{nullptr, 0, 0};
    PostingList* current = result_list;
    while (current) {
        if (results->num_results >= results->capacity) {
            results->capacity = (results->capacity == 0) ? 10 : results->capacity * 2;
            results->results = (SearchResult*)realloc(results->results, results->capacity * sizeof(SearchResult));
        }
        results->results[results->num_results++] = {current->doc_id, 1.0, nullptr};
        current = current->next;
    }
    return results;
}


// =================================================================
//  SECTION 2: PARALLEL SEARCH IMPLEMENTATION
// =================================================================

static PostingList* execute_node_parallel(InvertedIndex* index, QueryNode* node) {
    if (!node) return nullptr;
    if (node->left == nullptr && node->right == nullptr) return get_posting_list(index, node->term);
    PostingList* left_list = cilk_spawn execute_node_parallel(index, node->left);
    PostingList* right_list = execute_node_parallel(index, node->right);
    cilk_sync;
    if (node->op == 'A') return intersect_postings(left_list, right_list);
    if (node->op == 'O') return union_postings(left_list, right_list);
    return nullptr;
}

QueryNode* parse_query_to_tree(const std::string& query_string) {
    std::istringstream iss(query_string);
    std::vector<std::string> tokens;
    std::string token;
    while (iss >> token) tokens.push_back(token);
    if (tokens.empty()) return nullptr;

    // Use size_t to match the type of tokens.size() and avoid warnings
    size_t last_op_idx = std::string::npos;
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (tokens[i] == "AND" || tokens[i] == "OR") {
            last_op_idx = i;
        }
    }

    QueryNode* node = new QueryNode;
    if (last_op_idx == std::string::npos) { // No operators
        node->term = new char[tokens[0].length() + 1];
        strcpy(node->term, tokens[0].c_str());
        node->op = '\0'; node->left = node->right = nullptr;
    } else {
        node->term = nullptr; node->op = (tokens[last_op_idx] == "AND") ? 'A' : 'O';
        std::string left_q, right_q;
        for (size_t i = 0; i < last_op_idx; ++i) left_q += tokens[i] + " ";
        for (size_t i = last_op_idx + 1; i < tokens.size(); ++i) right_q += tokens[i] + " ";
        node->left = parse_query_to_tree(left_q);
        node->right = parse_query_to_tree(right_q);
    }
    return node;
}


ResultSet* execute_parallel_search(InvertedIndex* index, QueryNode* root) {
    if (!root) return nullptr;
    PostingList* final_list = execute_node_parallel(index, root);
    ResultSet* results = new ResultSet{nullptr, 0, 0};
    PostingList* current = final_list;
    while (current) {
        if (results->num_results >= results->capacity) {
            results->capacity = (results->capacity == 0) ? 10 : results->capacity * 2;
            results->results = (SearchResult*)realloc(results->results, results->capacity * sizeof(SearchResult));
        }
        results->results[results->num_results++] = {current->doc_id, 1.0, nullptr};
        current = current->next;
    }
    return results;
}

void free_query_tree(QueryNode* node) {
    if (!node) return;
    free_query_tree(node->left); free_query_tree(node->right);
    if (node->term) delete[] node->term;
    delete node;
}

// =================================================================
//  SECTION 3: SHARED UTILITY FUNCTIONS
// =================================================================

void free_result_set(ResultSet* results) {
    if (!results) return;
    if (results->results) {
        for(int i = 0; i < results->num_results; ++i) {
            if(results->results[i].snippet) delete[] results->results[i].snippet;
        }
        free(results->results);
    }
    delete results;
}