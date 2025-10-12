#ifndef QUERY_EXPANDER_H
#define QUERY_EXPANDER_H

#include "retrieval/query.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

/**
 * @class QueryExpander
 * @brief Transforms a simple user query string into a complex boolean query tree
 * by expanding keywords with a predefined set of synonyms loaded from a file.
 */
class QueryExpander {
public:
    /**
     * @brief Constructs the QueryExpander and loads synonyms from a specified file.
     * @param synonym_file_path Path to the synonym file.
     */
    explicit QueryExpander(const std::string& synonym_file_path);

    /**
     * @brief Expands a query string into a structured QueryNode tree.
     * @param query_str The raw query string from the user.
     * @return A unique_ptr to the root of the generated boolean query tree.
     */
    std::unique_ptr<QueryNode> expand_query(const std::string& query_str);

private:
    /**
     * @brief Loads and parses the synonym file.
     * @param synonym_file_path Path to the synonym file.
     */
    void load_synonyms(const std::string& synonym_file_path);

    // The in-memory map to store the loaded synonyms.
    std::unordered_map<std::string, std::vector<std::string>> synonym_map_;
};

#endif // QUERY_EXPANDER_H