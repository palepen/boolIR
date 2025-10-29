#ifndef QUERY_PREPROCESSOR_H
#define QUERY_PREPROCESSOR_H

#include <string>
#include <unordered_set>
#include <vector>

/**
 * Applies the same preprocessing to queries that was applied to documents:
 * - Lowercase conversion
 * - Punctuation removal
 * - Stop word removal
 * - Whitespace normalization
 */
class QueryPreprocessor
{
public:
    QueryPreprocessor();

    /**
     * Preprocess a query string to match document preprocessing
     */
    std::string preprocess(const std::string &query) const;

    /**
     * Load custom stop words from a file
     */
    void load_stop_words(const std::string &filepath);

    std::unordered_set<std::string> stop_words_;

    std::vector<std::string> tokenize(const std::string &text) const;

private:
    void initialize_default_stop_words();
    std::string to_lowercase(const std::string &text) const;
    std::string remove_punctuation(const std::string &text) const;
    std::string remove_stop_words(const std::vector<std::string> &tokens) const;
};

#endif // QUERY_PREPROCESSOR_H