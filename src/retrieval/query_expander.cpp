#include "retrieval/query_expander.h"
// REMOVED: #include "tokenizer/porter_stemmer.h"
#include <sstream>
#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <string>
#include <set>

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
        
        // Create OR node for this term and its synonyms
        auto or_node = std::make_unique<QueryNode>(QueryOperator::OR);
        
        // Use a set to avoid adding duplicate terms
        std::set<std::string> term_variations;
        
        // Always add the original term
        term_variations.insert(term);
        
        // Check for synonyms of the original term
        if (synonym_map_.count(term)) {
            for (const auto& synonym : synonym_map_.at(term)) {
                term_variations.insert(synonym);
            }
        }
        
        // Add all unique variations to the OR node
        for (const auto& variation : term_variations) {
            or_node->children.push_back(std::make_unique<QueryNode>(variation));
        }
        
        // If only one child in OR node (no synonyms found), just add it directly to root
        if (or_node->children.size() == 1) {
            root->children.push_back(std::move(or_node->children[0]));
        } else {
            root->children.push_back(std::move(or_node));
        }
    }
    
    return root;
}
