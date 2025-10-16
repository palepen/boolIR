#include "indexing/bsbi_indexer.h"
#include <cilk/cilk.h>
#include <cilk/cilk_stub.h>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <thread>
#include <filesystem>
#include <sstream>
#include <iomanip>

namespace fs = std::filesystem;

static std::vector<std::string> tokenize(const std::string &text)
{
    std::vector<std::string> tokens;
    std::stringstream ss(text);
    std::string token;
    while (ss >> token)
    {
        token.erase(std::remove_if(token.begin(), token.end(),
                                   [](unsigned char c)
                                   { return !std::isalnum(c); }),
                    token.end());
        if (!token.empty())
        {
            std::transform(token.begin(), token.end(), token.begin(),
                           [](unsigned char c)
                           { return std::tolower(c); });
            tokens.push_back(token);
        }
    }
    return tokens;
}

BSBIIndexer::BSBIIndexer(const DocumentCollection& documents,
                         const IdToDocNameMap& id_to_doc_name,  
                         const std::string& index_path,
                         const std::string& temp_path,
                         size_t block_size_mb,
                         size_t num_shards, 
                         size_t num_workers)
    : documents_(documents),
      id_to_doc_name_(id_to_doc_name),  
      index_path_(index_path),
      temp_path_(temp_path),
      block_size_bytes_(block_size_mb * 1024 * 1024),
      num_shards_(num_shards), 
      num_workers_(num_workers) {
    fs::create_directories(index_path_);
    fs::create_directories(temp_path_);
}

size_t BSBIIndexer::get_effective_workers() const {
    // If num_workers_ is 0, use all available cores
    // Otherwise, use the specified count
    return (num_workers_ == 0) ? std::thread::hardware_concurrency() : num_workers_;
}

void BSBIIndexer::build_index() {
    perf_monitor_.start_timer("Total Indexing Time");
    
    std::cout << "\n=== Starting BSBI Indexing (Sharded) ===" << std::endl;
    std::cout << "CPU workers: " << get_effective_workers() << std::endl;
    std::cout << "Shards to create: " << num_shards_ << std::endl;

    perf_monitor_.start_timer("Phase 1: Generate Runs");
    std::vector<std::string> run_files = generate_runs();
    perf_monitor_.end_timer("Phase 1: Generate Runs");

    perf_monitor_.start_timer("Phase 2: Merge Runs");
    std::string final_run_path = merge_runs(run_files);
    perf_monitor_.end_timer("Phase 2: Merge Runs");

    perf_monitor_.start_timer("Phase 3: Create Sharded Index");
    create_sharded_index_files(final_run_path); // MODIFIED
    perf_monitor_.end_timer("Phase 3: Create Sharded Index");
    
    perf_monitor_.start_timer("Phase 4: Create Document Store");
    create_document_store();
    perf_monitor_.end_timer("Phase 4: Create Document Store");
    
    fs::remove_all(temp_path_);
    perf_monitor_.end_timer("Total Indexing Time");
    print_indexing_summary();
}


// generate_runs() and merge_runs() remain the same as your optimized version.

void BSBIIndexer::create_sharded_index_files(const std::string &final_run_path) {
    std::cout << "\nPhase 3: Creating " << num_shards_ << " index shards..." << std::endl;

    for (size_t s = 0; s < num_shards_; ++s) {
        fs::create_directories(index_path_ + "/shard_" + std::to_string(s));
    }

    std::ifstream in(final_run_path, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot open final run file: " + final_run_path);

    std::vector<std::ofstream> shard_dicts(num_shards_);
    std::vector<std::ofstream> shard_postings(num_shards_);
    std::vector<long long> shard_offsets(num_shards_, 0);

    for (size_t s = 0; s < num_shards_; ++s) {
        shard_dicts[s].open(index_path_ + "/shard_" + std::to_string(s) + "/dict.dat", std::ios::binary);
        shard_postings[s].open(index_path_ + "/shard_" + std::to_string(s) + "/postings.dat", std::ios::binary);
    }

    std::string current_term;
    std::vector<unsigned int> postings;
    std::string term_str;
    unsigned int doc_id;

    while (std::getline(in, term_str, '\0') && in.read(reinterpret_cast<char *>(&doc_id), sizeof(doc_id))) {
        if (term_str != current_term && !current_term.empty()) {
            size_t shard_idx = std::hash<std::string>{}(current_term) % num_shards_;

            shard_dicts[shard_idx].write(current_term.c_str(), current_term.length() + 1);
            shard_dicts[shard_idx].write(reinterpret_cast<const char *>(&shard_offsets[shard_idx]), sizeof(long long));
            size_t list_size = postings.size();
            shard_dicts[shard_idx].write(reinterpret_cast<const char *>(&list_size), sizeof(size_t));
            shard_postings[shard_idx].write(reinterpret_cast<const char *>(postings.data()), list_size * sizeof(unsigned int));
            shard_offsets[shard_idx] = shard_postings[shard_idx].tellp();
            postings.clear();
        }
        current_term = term_str;
        postings.push_back(doc_id);
    }

    if (!current_term.empty()) {
        size_t shard_idx = std::hash<std::string>{}(current_term) % num_shards_;
        shard_dicts[shard_idx].write(current_term.c_str(), current_term.length() + 1);
        shard_dicts[shard_idx].write(reinterpret_cast<const char *>(&shard_offsets[shard_idx]), sizeof(long long));
        size_t list_size = postings.size();
        shard_dicts[shard_idx].write(reinterpret_cast<const char *>(&list_size), sizeof(size_t));
        shard_postings[shard_idx].write(reinterpret_cast<const char *>(postings.data()), list_size * sizeof(unsigned int));
    }
    
    std::cout << "  Sharded index created successfully." << std::endl;
}


void BSBIIndexer::print_indexing_summary() {
    double total_time = perf_monitor_.get_duration_ms("Total Indexing Time");
    double phase1_time = perf_monitor_.get_duration_ms("Phase 1: Generate Runs");
    double phase2_time = perf_monitor_.get_duration_ms("Phase 2: Merge Runs");
    double phase3_time = perf_monitor_.get_duration_ms("Phase 3: Create Final Index");
    double phase3_part_time = perf_monitor_.get_duration_ms("Phase 3: Create Partitioned Index");
    double phase4_time = perf_monitor_.get_duration_ms("Phase 4: Create Document Store");
    
    double phase3_actual = (phase3_time > 0) ? phase3_time : phase3_part_time;
    
    double throughput = (documents_.size() * 1000.0) / total_time;
    
    size_t effective_workers = get_effective_workers();
    
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "INDEXING PERFORMANCE SUMMARY" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    
    std::cout << "\nConfiguration:" << std::endl;
    std::cout << "  Total Documents: " << documents_.size() << std::endl;
    std::cout << "  CPU Workers: " << effective_workers << std::endl;  // Show actual
    std::cout << "  Block Size: " << (block_size_bytes_ / (1024 * 1024)) << " MB" << std::endl;
   
    
    std::cout << "\nOverall Performance:" << std::endl;
    std::cout << "  Total Time: " << std::fixed << std::setprecision(2) 
              << total_time << " ms (" << (total_time / 1000.0) << " seconds)" << std::endl;
    std::cout << "  Throughput: " << std::fixed << std::setprecision(0) 
              << throughput << " documents/second" << std::endl;
    
    std::cout << "\nPhase Breakdown:" << std::endl;
    std::cout << "  Phase                    | Time (ms) | Percentage" << std::endl;
    std::cout << "  -------------------------|-----------|------------" << std::endl;
    std::cout << "  1. Generate Runs         | " 
              << std::setw(9) << std::fixed << std::setprecision(0) << phase1_time << " | "
              << std::setw(9) << std::fixed << std::setprecision(1) << (phase1_time / total_time * 100.0) << "%" << std::endl;
    std::cout << "  2. Merge Runs            | " 
              << std::setw(9) << std::fixed << std::setprecision(0) << phase2_time << " | "
              << std::setw(9) << std::fixed << std::setprecision(1) << (phase2_time / total_time * 100.0) << "%" << std::endl;
    std::cout << "  3. Create Index          | " 
              << std::setw(9) << std::fixed << std::setprecision(0) << phase3_actual << " | "
              << std::setw(9) << std::fixed << std::setprecision(1) << (phase3_actual / total_time * 100.0) << "%" << std::endl;
    std::cout << "  4. Create Document Store | " 
              << std::setw(9) << std::fixed << std::setprecision(0) << phase4_time << " | "
              << std::setw(9) << std::fixed << std::setprecision(1) << (phase4_time / total_time * 100.0) << "%" << std::endl;
    
    std::cout << "\n" << std::string(70, '=') << std::endl;
}

std::vector<std::string> BSBIIndexer::generate_runs() {
    std::cout << "\nPhase 1: Generating sorted runs..." << std::endl;
    size_t num_docs = documents_.size();
    
    // Use configured worker count instead of hardware_concurrency()
    size_t effective_workers = get_effective_workers();
    size_t docs_per_worker = (num_docs + effective_workers - 1) / effective_workers;
    
    std::cout << "  Parallel processing with " << effective_workers << " workers" << std::endl;
    std::cout << "  ~" << docs_per_worker << " documents per worker" << std::endl;
    
    std::vector<std::string> all_run_files;

    cilk_for (size_t worker_id = 0; worker_id < effective_workers; ++worker_id) {
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

    std::cout << "  Generated " << all_run_files.size() << " initial run files" << std::endl;
    return all_run_files;
}

std::string BSBIIndexer::merge_runs(std::vector<std::string> &run_files)
{
    std::cout << "\nPhase 2: Merging runs..." << std::endl;
    int pass_num = 0;
    int total_merges = 0;

    while (run_files.size() > 1)
    {
        size_t files_to_merge = run_files.size();
        size_t pairs_to_merge = files_to_merge / 2;

        std::cout << "  Merge Pass " << ++pass_num << ": " << files_to_merge
                  << " files -> " << ((files_to_merge + 1) / 2) << " files" << std::endl;

        std::vector<std::string> next_pass_files;

        cilk_for(size_t i = 0; i < pairs_to_merge; ++i)
        {
            std::string file1_path = run_files[i * 2];
            std::string file2_path = run_files[i * 2 + 1];
            std::string out_path = temp_path_ + "/merge_p" + std::to_string(pass_num) + "_" + std::to_string(i) + ".dat";

            std::ifstream f1(file1_path, std::ios::binary);
            std::ifstream f2(file2_path, std::ios::binary);
            std::ofstream out(out_path, std::ios::binary);

            TermDocPair p1, p2;
            std::string term_str;
            bool f1_ok = std::getline(f1, term_str, '\0') && f1.read(reinterpret_cast<char *>(&p1.doc_id), sizeof(p1.doc_id));
            if (f1_ok)
                p1.term = term_str;
            bool f2_ok = std::getline(f2, term_str, '\0') && f2.read(reinterpret_cast<char *>(&p2.doc_id), sizeof(p2.doc_id));
            if (f2_ok)
                p2.term = term_str;

            while (f1_ok && f2_ok)
            {
                TermDocPair *smaller_pair = (p1 < p2) ? &p1 : &p2;
                out.write(smaller_pair->term.c_str(), smaller_pair->term.length() + 1);
                out.write(reinterpret_cast<const char *>(&smaller_pair->doc_id), sizeof(smaller_pair->doc_id));

                if (smaller_pair == &p1)
                {
                    f1_ok = std::getline(f1, term_str, '\0') && f1.read(reinterpret_cast<char *>(&p1.doc_id), sizeof(p1.doc_id));
                    if (f1_ok)
                        p1.term = term_str;
                }
                else
                {
                    f2_ok = std::getline(f2, term_str, '\0') && f2.read(reinterpret_cast<char *>(&p2.doc_id), sizeof(p2.doc_id));
                    if (f2_ok)
                        p2.term = term_str;
                }
            }

            while (f1_ok)
            {
                out.write(p1.term.c_str(), p1.term.length() + 1);
                out.write(reinterpret_cast<const char *>(&p1.doc_id), sizeof(p1.doc_id));
                f1_ok = std::getline(f1, term_str, '\0') && f1.read(reinterpret_cast<char *>(&p1.doc_id), sizeof(p1.doc_id));
                if (f1_ok)
                    p1.term = term_str;
            }
            while (f2_ok)
            {
                out.write(p2.term.c_str(), p2.term.length() + 1);
                out.write(reinterpret_cast<const char *>(&p2.doc_id), sizeof(p2.doc_id));
                f2_ok = std::getline(f2, term_str, '\0') && f2.read(reinterpret_cast<char *>(&p2.doc_id), sizeof(p2.doc_id));
                if (f2_ok)
                    p2.term = term_str;
            }
            {
                std::lock_guard<std::mutex> lock(vector_mutex_);
                next_pass_files.push_back(out_path);
            }
        }

        if (run_files.size() % 2 == 1)
        {
            next_pass_files.push_back(run_files.back());
        }

        total_merges += pairs_to_merge;
        run_files = next_pass_files;
    }

    std::cout << "  Merging complete: " << pass_num << " passes, "
              << total_merges << " total merge operations" << std::endl;
    return run_files.empty() ? "" : run_files[0];
}


void BSBIIndexer::create_document_store()
{
    std::cout << "\nPhase 4: Creating document store..." << std::endl;
    std::string doc_store_path = index_path_ + "/documents.dat";
    std::string doc_offset_path = index_path_ + "/doc_offsets.dat";
    std::string doc_names_path = index_path_ + "/doc_names.dat";  // NEW

    std::ofstream doc_store(doc_store_path, std::ios::binary);
    std::ofstream doc_offsets(doc_offset_path, std::ios::binary);
    std::ofstream doc_names(doc_names_path, std::ios::binary);  // NEW

    if (!doc_store || !doc_offsets || !doc_names)
    {
        throw std::runtime_error("Failed to create document store files");
    }

    long long current_offset = 0;

    for (const auto &doc : documents_)
    {
        // Write offset mapping: doc_id -> file_offset
        doc_offsets.write(reinterpret_cast<const char *>(&doc.id), sizeof(doc.id));
        doc_offsets.write(reinterpret_cast<const char *>(&current_offset), sizeof(current_offset));

        // Write document: doc_id, content_length, content
        uint32_t content_length = doc.content.length();
        doc_store.write(reinterpret_cast<const char *>(&doc.id), sizeof(doc.id));
        doc_store.write(reinterpret_cast<const char *>(&content_length), sizeof(content_length));
        doc_store.write(doc.content.c_str(), content_length);

        current_offset = doc_store.tellp();
        
        // NEW: Write document name mapping
        auto name_it = id_to_doc_name_.find(doc.id);
        if (name_it != id_to_doc_name_.end()) {
            const std::string& doc_name = name_it->second;
            uint32_t name_length = doc_name.length();
            
            doc_names.write(reinterpret_cast<const char*>(&doc.id), sizeof(doc.id));
            doc_names.write(reinterpret_cast<const char*>(&name_length), sizeof(name_length));
            doc_names.write(doc_name.c_str(), name_length);
        }
    }

    std::cout << "  Document store created: " << documents_.size() << " documents" << std::endl;
    std::cout << "  Store size: " << (current_offset / (1024.0 * 1024.0)) << " MB" << std::endl;
    std::cout << "  Document names saved: " << id_to_doc_name_.size() << " entries" << std::endl;
}