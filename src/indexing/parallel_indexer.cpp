#include "indexing/parallel_indexer.h"
#include <cilk/cilk.h>
#include <sstream>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <thread>
#include <unordered_map>

const std::unordered_map<std::string, PostingList> &ParallelIndexer::get_full_index() const
{
    if (!cache_is_valid_)
    {
        merged_index_cache_ = merge_shards();
        cache_is_valid_ = true;
    }

    return merged_index_cache_;
}

std::unordered_map<std::string, PostingList> ParallelIndexer::merge_shards() const
{
    std::unordered_map<std::string, PostingList> full_index;
    for (const auto &shard : shards_)
    {
        full_index.insert(shard.postings.begin(), shard.postings.end());
    }

    return full_index;
}

void ParallelIndexer::save_index(const std::string &filepath) const
{
    std::ofstream ofs(filepath, std::ios::binary);
    if (!ofs)
    {
        std::cerr << "Error: Could not open file for writing: " << filepath << std::endl;
        return;
    }
    const auto &full_index = get_full_index();
    size_t map_size = full_index.size();
    ofs.write(reinterpret_cast<const char *>(&map_size), sizeof(map_size));
    for (const auto &pair : full_index)
    {
        size_t term_len = pair.first.length();
        ofs.write(reinterpret_cast<const char *>(&term_len), sizeof(term_len));
        ofs.write(pair.first.c_str(), term_len);
        const auto &postings = pair.second.get_postings();
        size_t postings_size = postings.size();
        ofs.write(reinterpret_cast<const char *>(&postings_size), sizeof(postings_size));
        ofs.write(reinterpret_cast<const char *>(postings.data()), postings_size * sizeof(int));
    }
}

bool ParallelIndexer::load_index(const std::string &filepath) const
{
    std::ifstream ifs(filepath, std::ios::binary);
    if (!ifs)
    {
        std::cerr << "Error: Could not open file for reading: " << filepath << std::endl;
        return false;
    }
    merged_index_cache_.clear();
    size_t map_size;
    ifs.read(reinterpret_cast<char *>(&map_size), sizeof(map_size));
    for (size_t i = 0; i < map_size; ++i)
    {
        size_t term_len;
        ifs.read(reinterpret_cast<char *>(&term_len), sizeof(term_len));
        std::string term(term_len, '\0');
        ifs.read(&term[0], term_len);
        size_t postings_size;
        ifs.read(reinterpret_cast<char *>(&postings_size), sizeof(postings_size));
        std::vector<int> postings(postings_size);
        ifs.read(reinterpret_cast<char *>(postings.data()), postings_size * sizeof(int));
        PostingList pl;
        for (int doc_id : postings)
        {
            pl.add_document(doc_id);
        }
        merged_index_cache_[term] = pl;
    }
    cache_is_valid_ = true;
    return true;
}

// Simple tokenizer used by the parallel indexer.
static std::vector<std::string> tokenize_parallel(const std::string &text)
{
    std::vector<std::string> tokens;
    std::stringstream ss(text);
    std::string token;
    while (ss >> token)
    {
        for (char &c : token)
        {
            c = tolower(c);
        }
        tokens.push_back(token);
    }
    return tokens;
}

ParallelIndexer::ParallelIndexer(size_t num_shards) : shards_(num_shards), num_docs_indexed_(0) {}

size_t ParallelIndexer::hash_term_to_shard(const std::string &term) const
{
    return std::hash<std::string>{}(term) % shards_.size();
}

void ParallelIndexer::build_index_parallel(const DocumentCollection &documents)
{
    std::cout << "Starting parallel indexing for " << documents.size() << " documents with "
              << shards_.size() << " shards...\n";
    perf_monitor_.start_timer("total_indexing_time");

    num_docs_indexed_ = documents.size();

    // Determine chunk size for work distribution
    const size_t num_threads = std::thread::hardware_concurrency();
    const size_t chunk_size = (documents.size() + num_threads - 1) / num_threads;
    const size_t num_chunks = (documents.size() + chunk_size - 1) / chunk_size;

    // Store thread-local indices
    std::vector<std::unordered_map<std::string, std::vector<int>>> local_indices(num_chunks);

    // MAP Phase: Each thread builds its own complete local index
    cilk_for(size_t chunk_id = 0; chunk_id < num_chunks; ++chunk_id)
    {
        size_t start_idx = chunk_id * chunk_size;
        size_t end_idx = std::min(start_idx + chunk_size, documents.size());

        // Build local index for this chunk - NO LOCKS NEEDED
        auto &local_index = local_indices[chunk_id];

        for (size_t i = start_idx; i < end_idx; ++i)
        {
            const auto &doc = documents[i];
            auto tokens = tokenize_parallel(doc.content);

            // Track unique terms per document
            std::unordered_map<std::string, bool> seen_in_doc;

            for (const auto &token : tokens)
            {
                if (seen_in_doc.find(token) == seen_in_doc.end())
                {
                    seen_in_doc[token] = true;
                    local_index[token].push_back(doc.id);
                }
            }
        }
    }

    std::cout << "Map phase complete. Starting merge phase...\n";

    // REDUCE Phase: Merge local indices into shards
    // Process each shard independently in parallel
    cilk_for(size_t shard_id = 0; shard_id < shards_.size(); ++shard_id)
    {
        auto &shard = shards_[shard_id];

        // For each term that hashes to this shard
        for (size_t chunk_id = 0; chunk_id < num_chunks; ++chunk_id)
        {
            for (auto &term_pair : local_indices[chunk_id])
            {
                const std::string &term = term_pair.first;

                // Only process terms that belong to this shard
                if (hash_term_to_shard(term) == shard_id)
                {
                    auto &doc_ids = term_pair.second;

                    // No lock needed - each shard processed by single thread
                    auto &posting_list = shard.postings[term];
                    for (int doc_id : doc_ids)
                    {
                        posting_list.add_document(doc_id);
                    }
                }
            }
        }
    }

    perf_monitor_.end_timer("total_indexing_time");
    std::cout << "Parallel indexing complete.\n";
}

IndexingMetrics ParallelIndexer::get_performance_metrics() const
{
    IndexingMetrics metrics;
    double time_ms = perf_monitor_.get_duration_ms("total_indexing_time");
    metrics.indexing_time_ms = time_ms;
    metrics.throughput_docs_per_sec = (time_ms > 0) ? (1000.0 * num_docs_indexed_ / time_ms) : 0;
    return metrics;
}