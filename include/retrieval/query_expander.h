#ifndef QUERY_EXPANDER_H
#define QUERY_EXPANDER_H

#include "retrieval/query.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

/**
 * Transforms a simple user query string into a complex boolean query tree
 * by expanding keywords with a predefined set of synonyms loaded from a file.
 */
class QueryExpander {
public:
    /**
     * Constructs the QueryExpander and loads synonyms from a specified file.
     * @param synonym_file_path Path to the synonym file.
     */
    explicit QueryExpander(const std::string& synonym_file_path);

    /**
     * Expands a query string into a structured QueryNode tree.
     * @param query_str The raw query string from the user.
     * A unique_ptr to the root of the generated boolean query tree.
     */
    std::unique_ptr<QueryNode> expand_query(const std::string& query_str);

private:
    /**
     * Loads and parses the synonym file.
     * @param synonym_file_path Path to the synonym file.
     */
    void load_synonyms(const std::string& synonym_file_path);

    // The in-memory map to store the loaded synonyms.
    std::unordered_map<std::string, std::vector<std::string>> synonym_map_;
};

#endif // QUERY_EXPANDER_H