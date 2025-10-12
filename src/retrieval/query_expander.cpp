#include "retrieval/query_expander.h"
#include "tokenizer/porter_stemmer.h"
#include <sstream>
#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <string>

QueryExpander::QueryExpander(const std::string& synonym_file_path) {
    load_synonyms(synonym_file_path);
}

void QueryExpander::load_synonyms(const std::string& synonym_file_path) {
    std::ifstream file(synonym_file_path);
    if (!file.is_open()) {
        std::cerr << "Warning: Could not open synonym file: " << synonym_file_path 
                  << ". Query expansion will be disabled." << std::endl;
        return;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }

        std::stringstream ss(line);
        std::string term;
        std::string synonyms_str;

        if (std::getline(ss, term, ':') && std::getline(ss, synonyms_str)) {
            // Trim and convert term to lowercase
            term.erase(0, term.find_first_not_of(" \t\n\r\f\v"));
            term.erase(term.find_last_not_of(" \t\n\r\f\v") + 1);
            std::transform(term.begin(), term.end(), term.begin(), ::tolower);
            
            std::vector<std::string> synonym_list;
            std::stringstream syn_ss(synonyms_str);
            std::string synonym;

            while (std::getline(syn_ss, synonym, ',')) {
                synonym.erase(0, synonym.find_first_not_of(" \t\n\r\f\v"));
                synonym.erase(synonym.find_last_not_of(" \t\n\r\f\v") + 1);
                if (!synonym.empty()) {
                    std::transform(synonym.begin(), synonym.end(), synonym.begin(), ::tolower);
                    synonym_list.push_back(synonym);
                }
            }
            if (!synonym_list.empty()) {
                synonym_map_[term] = synonym_list;
            }
        }
    }
    std::cout << "Loaded " << synonym_map_.size() << " terms for query expansion from " 
              << synonym_file_path << std::endl;
}

std::unique_ptr<QueryNode> QueryExpander::expand_query(const std::string& query_str) {
    auto root = std::make_unique<QueryNode>(QueryOperator::AND);
    std::stringstream ss(query_str);
    std::string term;

    while (ss >> term) {
        // Remove punctuation
        term.erase(std::remove_if(term.begin(), term.end(), 
            [](unsigned char c) { return !std::isalnum(c); }), term.end());
        
        if (term.empty()) continue;
        
        // Convert to lowercase
        std::transform(term.begin(), term.end(), term.begin(), ::tolower);
        
        // Apply Porter stemming (matching the indexing process)
        std::string stemmed = PorterStemmer::stem(term);
        
        // Create OR node for this term's variations
        auto or_node = std::make_unique<QueryNode>(QueryOperator::OR);
        
        // Always add the stemmed version (this matches indexed terms)
        or_node->children.push_back(std::make_unique<QueryNode>(stemmed));
        
        // Check if original term (before stemming) has synonyms
        bool found_synonyms = false;
        if (synonym_map_.count(term)) {
            found_synonyms = true;
            for (const auto& synonym : synonym_map_.at(term)) {
                std::string syn_stemmed = PorterStemmer::stem(synonym);
                // Only add if different from main stemmed term
                if (syn_stemmed != stemmed) {
                    or_node->children.push_back(std::make_unique<QueryNode>(syn_stemmed));
                }
            }
        }
        
        // Also check if stemmed term has synonyms
        if (stemmed != term && synonym_map_.count(stemmed)) {
            for (const auto& synonym : synonym_map_.at(stemmed)) {
                std::string syn_stemmed = PorterStemmer::stem(synonym);
                if (syn_stemmed != stemmed) {
                    or_node->children.push_back(std::make_unique<QueryNode>(syn_stemmed));
                }
            }
        }
        
        // If only one child in OR node, just add it directly to root
        // This prevents unnecessary OR operations
        if (or_node->children.size() == 1) {
            root->children.push_back(std::move(or_node->children[0]));
        } else {
            root->children.push_back(std::move(or_node));
        }
    }
    
    return root;
}