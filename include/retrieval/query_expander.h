#ifndef QUERY_EXPANDER_H
#define QUERY_EXPANDER_H

#include "retrieval/query.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

/**
 * Transforms a simple user query string into a complex boolean query tree
 * by parsing boolean operators (AND, OR, NOT) and expanding keywords
 * with a predefined set of synonyms loaded from a file.
 */
class QueryExpander {
public:
    explicit QueryExpander(const std::string& synonym_file_path);

    std::unique_ptr<QueryNode> expand_query(const std::string& query_str);

private:
    /**
     * Loads and parses the synonym file.
     */
    void load_synonyms(const std::string& synonym_file_path);

    // The in-memory map to store the loaded synonyms.
    std::unordered_map<std::string, std::vector<std::string>> synonym_map_;

    std::vector<std::string> tokens_;
    size_t current_token_index_;

    /**
     * Creates an OR-node for a term and all its synonyms.
     */
    std::unique_ptr<QueryNode> create_synonym_node(const std::string& term);

    /**
     * Parses an expression (handles OR, lowest precedence).
     */
    std::unique_ptr<QueryNode> parse_expression();

    /**
     * Parses a term (handles AND / implicit AND, middle precedence).
     */
    std::unique_ptr<QueryNode> parse_term();

    /**
     * Parses a factor (handles NOT, parentheses, and TERM, highest precedence).
     */
    std::unique_ptr<QueryNode> parse_factor();

    bool is_at_end() const;
    std::string peek() const;
    std::string consume();
};

#endif // QUERY_EXPANDER_H