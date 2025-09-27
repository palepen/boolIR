#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <dirent.h>
#include <algorithm>
#include <cctype>
#include <mutex>
#include <cilk/cilk.h>
#include "indexing.h"

// Define a mutex for thread-safe access to the inverted index
std::mutex index_mutex;

// Helper function to process and tokenize a line of text
void process_line(const std::string& line, InvertedIndex* index, uint32_t doc_id) {
    std::string current_token;
    for (char ch : line) {
        if (std::isalnum(ch)) {
            current_token += std::tolower(ch);
        } else {
            if (!current_token.empty()) {
                // Lock the mutex before modifying the shared index
                std::lock_guard<std::mutex> lock(index_mutex);
                
                // Find the term in the index or add it
                auto it = index->postings.find(current_token);
                if (it == index->postings.end()) {
                    // Term not found, create a new posting list
                    index->postings[current_token] = new PostingList{doc_id, 1, nullptr};
                } else {
                    // Term found, add to its posting list
                    PostingList* current = it->second;
                    while (current->next != nullptr && current->doc_id != doc_id) {
                        current = current->next;
                    }
                    if (current->doc_id == doc_id) {
                        current->term_freq++;
                    } else {
                        current->next = new PostingList{doc_id, 1, nullptr};
                    }
                }
                current_token.clear();
            }
        }
    }
    // Add the last token if it exists
    if (!current_token.empty()) {
        std::lock_guard<std::mutex> lock(index_mutex);
        auto it = index->postings.find(current_token);
        if (it == index->postings.end()) {
            index->postings[current_token] = new PostingList{doc_id, 1, nullptr};
        } else {
             PostingList* current = it->second;
            while (current->next != nullptr && current->doc_id != doc_id) {
                current = current->next;
            }
            if (current->doc_id == doc_id) {
                current->term_freq++;
            } else {
                current->next = new PostingList{doc_id, 1, nullptr};
            }
        }
    }
}


// Process a single file and add its content to the index
void process_file(const std::string& file_path, InvertedIndex* index, uint32_t doc_id) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        std::cerr << "Could not open file: " << file_path << std::endl;
        return;
    }

    std::string line;
    while (std::getline(file, line)) {
        process_line(line, index, doc_id);
    }
}


// C-style interface functions exposed to the rest of the application

extern "C" {

InvertedIndex* create_inverted_index(void) {
    return new InvertedIndex;
}

void free_inverted_index(InvertedIndex* index) {
    if (!index) return;
    for (auto const& [term, list] : index->postings) {
        PostingList* current = list;
        while (current != nullptr) {
            PostingList* next = current->next;
            delete current;
            current = next;
        }
    }
    delete index;
}

void build_index_parallel(InvertedIndex* index, const char* dataset_path) {
    DIR* dir = opendir(dataset_path);
    if (!dir) {
        perror("Could not open dataset directory");
        return;
    }

    std::vector<std::string> file_paths;
    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        if (entry->d_type == DT_REG) {
            file_paths.push_back(std::string(dataset_path) + "/" + std::string(entry->d_name));
        }
    }
    closedir(dir);

    // Process files in parallel using OpenCilk
    cilk_for (int i = 0; i < file_paths.size(); ++i) {
        process_file(file_paths[i], index, i);
    }
}

PostingList* get_posting_list(InvertedIndex* index, const char* term) {
    std::string term_str(term);
    auto it = index->postings.find(term_str);
    if (it != index->postings.end()) {
        return it->second;
    }
    return nullptr;
}

}