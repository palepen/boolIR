#include "indexing/bsbi_indexer.h"
#include "tokenizer/porter_stemmer.h"
#include <cilk/cilk.h>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <thread>
#include <filesystem>
#include <sstream>

namespace fs = std::filesystem;

static std::vector<std::string> tokenize(const std::string& text) {
    std::vector<std::string> tokens;
    std::stringstream ss(text);
    std::string token;
    while (ss >> token) {
        token.erase(std::remove_if(token.begin(), token.end(), 
            [](unsigned char c) { return !std::isalnum(c); }), token.end());
        if (!token.empty()) {
            std::transform(token.begin(), token.end(), token.begin(),
                           [](unsigned char c){ return std::tolower(c); });
            // Apply Porter stemming
            token = PorterStemmer::stem(token);
            tokens.push_back(token);
        }
    }
    return tokens;
}

BSBIIndexer::BSBIIndexer(const DocumentCollection& documents,
                         const std::string& index_path,
                         const std::string& temp_path,
                         size_t block_size_mb)
    : documents_(documents),
      index_path_(index_path),
      temp_path_(temp_path),
      block_size_bytes_(block_size_mb * 1024 * 1024) {
    fs::create_directories(index_path_);
    fs::create_directories(temp_path_);
}

void BSBIIndexer::build_index() {
    perf_monitor_.start_timer("Total Indexing Time");
    std::cout << "Starting BSBI Indexing with Porter Stemming..." << std::endl;
    std::cout << "  -> Block size: " << (block_size_bytes_ / (1024 * 1024)) << " MB" << std::endl;

    perf_monitor_.start_timer("Phase 1: Generate Runs");
    std::vector<std::string> run_files = generate_runs();
    perf_monitor_.end_timer("Phase 1: Generate Runs");

    perf_monitor_.start_timer("Phase 2: Merge Runs");
    std::string final_run_path = merge_runs(run_files);
    perf_monitor_.end_timer("Phase 2: Merge Runs");

    perf_monitor_.start_timer("Phase 3: Create Final Index");
    create_final_index_files(final_run_path);
    perf_monitor_.end_timer("Phase 3: Create Final Index");
    
    fs::remove_all(temp_path_);

    perf_monitor_.end_timer("Total Indexing Time");
    perf_monitor_.print_summary();
}



std::vector<std::string> BSBIIndexer::generate_runs() {
    std::cout << "Phase 1: Generating sorted runs with Porter stemming..." << std::endl;
    size_t num_docs = documents_.size();
    size_t num_workers = std::thread::hardware_concurrency();
    size_t docs_per_worker = (num_docs + num_workers - 1) / num_workers;
    
    std::vector<std::string> all_run_files;

    cilk_for (size_t worker_id = 0; worker_id < num_workers; ++worker_id) {
        size_t start_doc = worker_id * docs_per_worker;
        size_t end_doc = std::min(start_doc + docs_per_worker, num_docs);
        
        std::vector<TermDocPair> buffer;
        size_t current_buffer_size = 0;
        int block_num = 0;

        for (size_t i = start_doc; i < end_doc; ++i) {
            const auto& doc = documents_[i];
            auto tokens = tokenize(doc.content);

            for (const auto& token : tokens) {
                buffer.push_back({token, doc.id});
                current_buffer_size += sizeof(doc.id) + token.length() + 1;
            }

            if (current_buffer_size >= block_size_bytes_) {
                std::sort(buffer.begin(), buffer.end());
                std::string run_path = temp_path_ + "/run_w" + std::to_string(worker_id) + "_b" + std::to_string(block_num++) + ".dat";
                
                std::ofstream out(run_path, std::ios::binary);
                for (const auto& pair : buffer) {
                    out.write(pair.term.c_str(), pair.term.length() + 1);
                    out.write(reinterpret_cast<const char*>(&pair.doc_id), sizeof(pair.doc_id));
                }

                { std::lock_guard<std::mutex> lock(vector_mutex_); all_run_files.push_back(run_path); }
                buffer.clear();
                current_buffer_size = 0;
            }
        }

        if (!buffer.empty()) {
            std::sort(buffer.begin(), buffer.end());
            std::string run_path = temp_path_ + "/run_w" + std::to_string(worker_id) + "_b" + std::to_string(block_num++) + ".dat";
            std::ofstream out(run_path, std::ios::binary);
            for (const auto& pair : buffer) {
                out.write(pair.term.c_str(), pair.term.length() + 1);
                out.write(reinterpret_cast<const char*>(&pair.doc_id), sizeof(pair.doc_id));
            }
            { std::lock_guard<std::mutex> lock(vector_mutex_); all_run_files.push_back(run_path); }
        }
    }

    std::cout << "  -> Generated " << all_run_files.size() << " initial run files." << std::endl;
    return all_run_files;
}

std::string BSBIIndexer::merge_runs(std::vector<std::string>& run_files) {
    std::cout << "Phase 2: Merging runs..." << std::endl;
    int pass_num = 0;
    while (run_files.size() > 1) {
        std::cout << "  -> Merge Pass " << ++pass_num << ": merging " << run_files.size() << " files into " << (run_files.size() + 1) / 2 << "..." << std::endl;
        std::vector<std::string> next_pass_files;
        
        cilk_for (size_t i = 0; i < run_files.size() / 2; ++i) {
            std::string file1_path = run_files[i * 2];
            std::string file2_path = run_files[i * 2 + 1];
            std::string out_path = temp_path_ + "/merge_p" + std::to_string(pass_num) + "_" + std::to_string(i) + ".dat";
            
            std::ifstream f1(file1_path, std::ios::binary);
            std::ifstream f2(file2_path, std::ios::binary);
            std::ofstream out(out_path, std::ios::binary);

            TermDocPair p1, p2;
            std::string term_str;
            bool f1_ok = std::getline(f1, term_str, '\0') && f1.read(reinterpret_cast<char*>(&p1.doc_id), sizeof(p1.doc_id));
            if(f1_ok) p1.term = term_str;
            bool f2_ok = std::getline(f2, term_str, '\0') && f2.read(reinterpret_cast<char*>(&p2.doc_id), sizeof(p2.doc_id));
            if(f2_ok) p2.term = term_str;

            while (f1_ok && f2_ok) {
                TermDocPair* smaller_pair = (p1 < p2) ? &p1 : &p2;
                out.write(smaller_pair->term.c_str(), smaller_pair->term.length() + 1);
                out.write(reinterpret_cast<const char*>(&smaller_pair->doc_id), sizeof(smaller_pair->doc_id));
                
                if (smaller_pair == &p1) {
                    f1_ok = std::getline(f1, term_str, '\0') && f1.read(reinterpret_cast<char*>(&p1.doc_id), sizeof(p1.doc_id));
                    if(f1_ok) p1.term = term_str;
                } else {
                    f2_ok = std::getline(f2, term_str, '\0') && f2.read(reinterpret_cast<char*>(&p2.doc_id), sizeof(p2.doc_id));
                    if(f2_ok) p2.term = term_str;
                }
            }

            while (f1_ok) {
                 out.write(p1.term.c_str(), p1.term.length() + 1);
                 out.write(reinterpret_cast<const char*>(&p1.doc_id), sizeof(p1.doc_id));
                 f1_ok = std::getline(f1, term_str, '\0') && f1.read(reinterpret_cast<char*>(&p1.doc_id), sizeof(p1.doc_id));
                 if(f1_ok) p1.term = term_str;
            }
            while (f2_ok) {
                 out.write(p2.term.c_str(), p2.term.length() + 1);
                 out.write(reinterpret_cast<const char*>(&p2.doc_id), sizeof(p2.doc_id));
                 f2_ok = std::getline(f2, term_str, '\0') && f2.read(reinterpret_cast<char*>(&p2.doc_id), sizeof(p2.doc_id));
                 if(f2_ok) p2.term = term_str;
            }
            { std::lock_guard<std::mutex> lock(vector_mutex_); next_pass_files.push_back(out_path); }
        }

        if (run_files.size() % 2 == 1) {
            next_pass_files.push_back(run_files.back());
        }

        run_files = next_pass_files;
    }
    
    std::cout << "  -> Merging complete." << std::endl;
    return run_files.empty() ? "" : run_files[0];
}

void BSBIIndexer::create_final_index_files(const std::string& final_run_path) {
    std::cout << "Phase 3: Creating final dictionary and postings files..." << std::endl;
    std::ifstream in(final_run_path, std::ios::binary);
    std::ofstream dict_out(index_path_ + "/dictionary.dat", std::ios::binary);
    std::ofstream post_out(index_path_ + "/postings.dat", std::ios::binary);
    
    std::string current_term;
    std::vector<unsigned int> postings;
    long long current_offset = 0;
    
    std::string term_str;
    unsigned int doc_id;
    while (std::getline(in, term_str, '\0') && in.read(reinterpret_cast<char*>(&doc_id), sizeof(doc_id))) {
        if (term_str != current_term && !current_term.empty()) {
            dict_out.write(current_term.c_str(), current_term.length() + 1);
            dict_out.write(reinterpret_cast<const char*>(&current_offset), sizeof(current_offset));
            size_t list_size = postings.size();
            dict_out.write(reinterpret_cast<const char*>(&list_size), sizeof(list_size));

            post_out.write(reinterpret_cast<const char*>(postings.data()), list_size * sizeof(unsigned int));
            current_offset = post_out.tellp();
            
            postings.clear();
        }
        current_term = term_str;
        if (postings.empty() || postings.back() != doc_id) {
            postings.push_back(doc_id);
        }
    }

    if (!current_term.empty()) {
        dict_out.write(current_term.c_str(), current_term.length() + 1);
        dict_out.write(reinterpret_cast<const char*>(&current_offset), sizeof(current_offset));
        size_t list_size = postings.size();
        dict_out.write(reinterpret_cast<const char*>(&list_size), sizeof(list_size));
        post_out.write(reinterpret_cast<const char*>(postings.data()), list_size * sizeof(unsigned int));
    }

    std::cout << "  -> Final index created in: " << index_path_ << std::endl;
}