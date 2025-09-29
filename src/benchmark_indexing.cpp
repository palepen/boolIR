#include "indexing/sequential_indexer.h"
#include "indexing/parallel_indexer.h"
#include <iostream>
#include <iomanip>
#include <vector>

// Generates a sample collection of documents for benchmarking.
DocumentCollection create_sample_documents(int num_docs) {
    DocumentCollection docs;
    docs.reserve(num_docs);
    for (int i = 0; i < num_docs; ++i) {
        docs.push_back({i, "the quick brown fox jumps over the lazy dog document " + std::to_string(i)});
    }
    return docs;
}

int main() {
    const int num_docs = 50000;
    const uint16_t num_shards = 32;
    std::cout << "Preparing " << num_docs << " sample documents for benchmark...\n\n";
    DocumentCollection documents = create_sample_documents(num_docs);

    // --- Benchmark Sequential Indexer ---
    SequentialIndexer seq_indexer;
    seq_indexer.build_index(documents);
    IndexingMetrics seq_metrics = seq_indexer.get_performance_metrics();

    // --- Benchmark Parallel Indexer ---
    // Use 16 shards as specified in the guide.
    ParallelIndexer par_indexer(num_shards); 
    par_indexer.build_index_parallel(documents);
    IndexingMetrics par_metrics = par_indexer.get_performance_metrics();

    // --- Print Final Comparison Report ---
    std::cout << "\n\n--- Indexing Benchmark Results (" << num_docs << " documents) ---\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "------------------------------------------------------------\n";
    std::cout << std::left << std::setw(25) << "Indexer" << std::setw(20) << "Time (ms)" << std::setw(25) << "Throughput (docs/s)" << "\n";
    std::cout << "------------------------------------------------------------\n";
    
    std::cout << std::left << std::setw(25) << "Sequential" 
              << std::setw(20) << seq_metrics.indexing_time_ms 
              << std::setw(25) << seq_metrics.throughput_docs_per_sec << "\n";

    std::cout << std::left << std::setw(25) << "Parallel (" << num_shards << " shards)" 
              << std::setw(20) << par_metrics.indexing_time_ms 
              << std::setw(25) << par_metrics.throughput_docs_per_sec << "\n";
    
    std::cout << "------------------------------------------------------------\n";

    double speedup = (seq_metrics.indexing_time_ms > 0) ? (seq_metrics.indexing_time_ms / par_metrics.indexing_time_ms) : 0.0;
    std::cout << "\nSpeedup Factor: " << speedup << "x\n\n";

    return 0;
}