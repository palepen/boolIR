#ifndef QUERY_PREPROCESSOR_H
#define QUERY_PREPROCESSOR_H

#include <string>
#include <unordered_set>
#include <vector>

/**
 * @class QueryPreprocessor
 * @brief Handles query normalization and preprocessing for consistent matching
 * 
 * Applies the same preprocessing to queries that was applied to documents:
 * - Lowercase conversion
 * - Punctuation removal
 * - Stop word removal
 * - Whitespace normalization
 */
class QueryPreprocessor {
public:
    QueryPreprocessor();
    
    /**
     * @brief Preprocess a query string to match document preprocessing
     * @param query Raw query text
     * @return Preprocessed query text
     */
    std::string preprocess(const std::string& query) const;
    
    /**
     * @brief Load custom stop words from a file
     * @param filepath Path to stop words file (one word per line)
     */
    void load_stop_words(const std::string& filepath);
    
private:
    std::unordered_set<std::string> stop_words_;
    
    void initialize_default_stop_words();
    std::string to_lowercase(const std::string& text) const;
    std::string remove_punctuation(const std::string& text) const;
    std::vector<std::string> tokenize(const std::string& text) const;
    std::string remove_stop_words(const std::vector<std::string>& tokens) const;
};

#endif // QUERY_PREPROCESSOR_H