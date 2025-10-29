#include "retrieval/query_preprocessor.h"
#include <algorithm>
#include <cctype>
#include <sstream>
#include <fstream>
#include <iostream>

QueryPreprocessor::QueryPreprocessor() {
    initialize_default_stop_words();
}

void QueryPreprocessor::initialize_default_stop_words() {
    // Common English stop words
    // MODIFIED: "and" has been REMOVED from this list to allow boolean parsing
    std::vector<std::string> default_stops = {
        "a", "an", "are", "as", "at", "be", "by", "for", "from",
        "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
        "to", "was", "will", "with", "what", "when", "where", "who", "how",
        "which", "this", "these", "those", "can", "could", "do", "does",
        "have", "had", "been", "being", "would", "should", "may", "might"
    };
    
    for (const auto& word : default_stops) {
        stop_words_.insert(word);
    }
    
    std::cout << "Query preprocessor initialized with " << stop_words_.size() 
              << " stop words" << std::endl;
}

void QueryPreprocessor::load_stop_words(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Warning: Could not open stop words file: " << filepath << std::endl;
        return;
    }
    
    stop_words_.clear();
    std::string word;
    while (std::getline(file, word)) {
        word.erase(0, word.find_first_not_of(" \t\n\r\f\v"));
        word.erase(word.find_last_not_of(" \t\n\r\f\v") + 1);
        if (!word.empty() && word[0] != '#') {
            // Also ensure "and" is not added if it's in a custom file
            std::string lower_word = to_lowercase(word);
            if (lower_word != "and" && lower_word != "or" && lower_word != "not") {
                 stop_words_.insert(lower_word);
            }
        }
    }
    
    std::cout << "Loaded " << stop_words_.size() << " custom stop words from " 
              << filepath << std::endl;
}

std::string QueryPreprocessor::to_lowercase(const std::string& text) const {
    std::string result = text;
    std::transform(result.begin(), result.end(), result.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return result;
}

std::string QueryPreprocessor::remove_punctuation(const std::string& text) const {
    std::string result;
    result.reserve(text.length());
    
    for (char c : text) {
        // Allow parentheses for grouping
        if (std::isalnum(static_cast<unsigned char>(c)) || std::isspace(static_cast<unsigned char>(c)) || c == '(' || c == ')') {
            result += c;
        } else {
            result += ' ';  // Replace other punctuation with space
        }
    }
    return result;
}

std::vector<std::string> QueryPreprocessor::tokenize(const std::string& text) const {
    std::vector<std::string> tokens;
    std::stringstream ss(text);
    std::string token;
    
    while (ss >> token) {
        if (!token.empty()) {
            tokens.push_back(token);
        }
    }
    return tokens;
}

std::string QueryPreprocessor::remove_stop_words(const std::vector<std::string>& tokens) const {
    std::ostringstream result;
    bool first = true;
    
    for (const auto& token : tokens) {
        if (stop_words_.find(token) == stop_words_.end()) {
            if (!first) result << " ";
            result << token;
            first = false;
        }
    }
    return result.str();
}

std::string QueryPreprocessor::preprocess(const std::string& query) const {
    // Step 1: Convert to lowercase
    std::string processed = to_lowercase(query);
    
    // Step 2: Remove punctuation (but keep parentheses)
    processed = remove_punctuation(processed);
    
    // Step 3: Tokenize
    auto tokens = tokenize(processed);
    
    // Step 4: Remove stop words
    processed = remove_stop_words(tokens);
    
    // Step 5: Trim whitespace
    processed.erase(0, processed.find_first_not_of(" \t\n\r\f\v"));
    processed.erase(processed.find_last_not_of(" \t\n\r\f\v") + 1);
    
    return processed;
}