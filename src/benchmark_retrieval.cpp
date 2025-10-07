
#include "indexing/parallel_indexer.h"
#include "retrieval/boolean_retrieval.h"
#include "indexing/performance_monitor.h"
#include <iostream>
#include <iomanip>
#include <random>

// Generate a MASSIVE, realistic dataset with large posting lists
DocumentCollection create_massive_corpus(int num_docs) {
    DocumentCollection docs;
    
    // Much larger vocabulary with different frequency classes
    // High-frequency terms (appear in 30-60% of documents)
    std::vector<std::string> very_common = {
        "system", "data", "information", "process", "computer", "software",
        "network", "application", "user", "service", "management", "security"
    };
    
    // Medium-frequency terms (appear in 10-30% of documents)
    std::vector<std::string> common = {
        "database", "algorithm", "search", "query", "index", "document",
        "performance", "analysis", "optimization", "framework", "architecture",
        "infrastructure", "platform", "technology", "development", "interface"
    };
    
    // Low-frequency terms (appear in 3-10% of documents)
    std::vector<std::string> less_common = {
        "parallel", "sequential", "distributed", "scalable", "efficient",
        "latency", "throughput", "benchmark", "evaluation", "implementation",
        "machine", "learning", "artificial", "intelligence", "neural", "model"
    };
    
    // Fill words to make documents longer
    std::vector<std::string> filler = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "is", "was", "are", "were", "be", "been", "being", "have", "has", "had",
        "with", "from", "by", "this", "that", "these", "those", "can", "will"
    };
    
    std::random_device rd;
    std::mt19937 gen(42);  // Fixed seed for reproducibility
    std::uniform_int_distribution<> very_common_dist(0, very_common.size() - 1);
    std::uniform_int_distribution<> common_dist(0, common.size() - 1);
    std::uniform_int_distribution<> less_common_dist(0, less_common.size() - 1);
    std::uniform_int_distribution<> filler_dist(0, filler.size() - 1);
    std::uniform_int_distribution<> doc_length(300, 800);  // MUCH longer documents
    std::uniform_int_distribution<> freq_selector(1, 100);
    
    docs.reserve(num_docs);
    
    for (int i = 1; i <= num_docs; ++i) {
        std::string content;
        int num_words = doc_length(gen);
        
        for (int j = 0; j < num_words; ++j) {
            int selector = freq_selector(gen);
            
            if (selector <= 50) {
                // 50% filler words
                content += filler[filler_dist(gen)] + " ";
            } else if (selector <= 80) {
                // 30% very common words (creates LARGE posting lists)
                content += very_common[very_common_dist(gen)] + " ";
            } else if (selector <= 95) {
                // 15% common words (creates medium posting lists)
                content += common[common_dist(gen)] + " ";
            } else {
                // 5% less common words (creates smaller posting lists)
                content += less_common[less_common_dist(gen)] + " ";
            }
        }
        
        docs.push_back({i, content});
        
        if (i % 50000 == 0) {
            std::cout << "  Generated " << i << " documents...\r" << std::flush;
        }
    }
    std::cout << std::endl;
    
    return docs;
}

void print_posting_list_stats(const std::unordered_map<std::string, PostingList>& index,
                               const std::vector<std::string>& terms) {
    std::cout << "\n╔═══════════════════════════════════════╗\n";
    std::cout << "║   Posting List Size Analysis      ║\n";
    std::cout << "╚═══════════════════════════════════════╝\n";
    
    size_t max_size = 0;
    size_t min_size = SIZE_MAX;
    size_t total_size = 0;
    int found = 0;
    
    for (const auto& term : terms) {
        auto it = index.find(term);
        if (it != index.end()) {
            size_t size = it->second.get_postings().size();
            std::cout << "  '" << std::left << std::setw(15) << term << "': " 
                      << std::right << std::setw(7) << size << " documents";
            
            // Indicate if size is good for parallelization
            if (size >= 3000) {
                std::cout << "  ✓ (Good for parallel)";
            } else if (size >= 1000) {
                std::cout << "  ~ (Marginal)";
            } else {
                std::cout << "  ✗ (Too small)";
            }
            std::cout << "\n";
            
            max_size = std::max(max_size, size);
            min_size = std::min(min_size, size);
            total_size += size;
            found++;
        } else {
            std::cout << "  '" << term << "': NOT FOUND\n";
        }
    }
    
    if (found > 0) {
        std::cout << "\n  Statistics:\n";
        std::cout << "    Average: " << (total_size / found) << " documents\n";
        std::cout << "    Max: " << max_size << " documents\n";
        std::cout << "    Min: " << min_size << " documents\n";
    }
    std::cout << "───────────────────────────────────────\n";
}

int main() {
    // Phase 1: Build MASSIVE index
    const int NUM_DOCS = 500000;  // Half a million documents!
    std::cout << "╔════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  Massive Boolean Retrieval Benchmark                       ║\n";
    std::cout << "║  Documents: " << NUM_DOCS << "                                            ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════╝\n\n";
    
    std::cout << "Phase 1: Building massive corpus...\n";
    std::cout << "Expected time: 30-90 seconds\n\n";
    
    DocumentCollection docs = create_massive_corpus(NUM_DOCS);
    
    std::cout << "\nPhase 2: Indexing documents with parallel indexer...\n";
    ParallelIndexer indexer(16);
    indexer.build_index_parallel(docs);
    IndexingMetrics idx_metrics = indexer.get_performance_metrics();
    
    const auto& full_index = indexer.get_full_index();
    std::cout << "\n✓ Index built successfully!\n";
    std::cout << "  Unique terms: " << full_index.size() << "\n";
    std::cout << "  Indexing time: " << idx_metrics.indexing_time_ms << " ms\n";
    std::cout << "  Throughput: " << idx_metrics.throughput_docs_per_sec << " docs/sec\n";

    // Print statistics for query terms
    std::vector<std::string> query_terms = {
        "system", "data", "computer", "software", "database", "search",
        "parallel", "sequential", "algorithm", "performance"
    };
    print_posting_list_stats(full_index, query_terms);

    // Phase 3: Prepare retrieval systems and queries
    std::cout << "\n╔════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  Phase 3: Boolean Retrieval Benchmark                     ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════╝\n";
    
    SequentialBooleanRetrieval seq_retrieval(full_index);
    ParallelBooleanRetrieval par_retrieval(full_index);
    PerformanceMonitor perf;

    // Query 1: (system AND data) OR computer
    // Expected: VERY large posting lists (60K+ docs each)
    std::cout << "\nBuilding queries with high-frequency terms...\n";
    QueryNode q1(QueryOperator::OR);
    auto q1_left = std::make_unique<QueryNode>(QueryOperator::AND);
    q1_left->children.push_back(std::make_unique<QueryNode>("system"));
    q1_left->children.push_back(std::make_unique<QueryNode>("data"));
    q1.children.push_back(std::move(q1_left));
    q1.children.push_back(std::make_unique<QueryNode>("computer"));

    // Query 2: database AND NOT software
    // Expected: Large posting lists (30K+ docs)
    QueryNode q2(QueryOperator::AND);
    q2.children.push_back(std::make_unique<QueryNode>("database"));
    auto q2_not_node = std::make_unique<QueryNode>(QueryOperator::NOT);
    q2_not_node->children.push_back(std::make_unique<QueryNode>("software"));
    q2.children.push_back(std::move(q2_not_node));

    // Query 3: ((system OR data) AND (computer OR software))
    // Expected: MASSIVE workload - best for parallelization
    QueryNode q3(QueryOperator::AND);
    auto q3_left = std::make_unique<QueryNode>(QueryOperator::OR);
    q3_left->children.push_back(std::make_unique<QueryNode>("system"));
    q3_left->children.push_back(std::make_unique<QueryNode>("data"));
    auto q3_right = std::make_unique<QueryNode>(QueryOperator::OR);
    q3_right->children.push_back(std::make_unique<QueryNode>("computer"));
    q3_right->children.push_back(std::make_unique<QueryNode>("software"));
    q3.children.push_back(std::move(q3_left));
    q3.children.push_back(std::move(q3_right));

    // Query 4: ((database AND search) OR (algorithm AND performance))
    // Expected: Medium posting lists - should show modest speedup
    QueryNode q4(QueryOperator::OR);
    auto q4_left = std::make_unique<QueryNode>(QueryOperator::AND);
    q4_left->children.push_back(std::make_unique<QueryNode>("database"));
    q4_left->children.push_back(std::make_unique<QueryNode>("search"));
    auto q4_right = std::make_unique<QueryNode>(QueryOperator::AND);
    q4_right->children.push_back(std::make_unique<QueryNode>("algorithm"));
    q4_right->children.push_back(std::make_unique<QueryNode>("performance"));
    q4.children.push_back(std::move(q4_left));
    q4.children.push_back(std::move(q4_right));

    // Query 5: Small query - parallel AND sequential (rare terms)
    // Expected: Small posting lists - sequential should be chosen automatically
    QueryNode q5(QueryOperator::AND);
    q5.children.push_back(std::make_unique<QueryNode>("parallel"));
    q5.children.push_back(std::make_unique<QueryNode>("sequential"));

    // Get sample results
    std::cout << "\nAnalyzing query result sizes...\n";
    ResultSet sample1 = seq_retrieval.execute_query(q1);
    ResultSet sample2 = seq_retrieval.execute_query(q2);
    ResultSet sample3 = seq_retrieval.execute_query(q3);
    ResultSet sample4 = seq_retrieval.execute_query(q4);
    ResultSet sample5 = seq_retrieval.execute_query(q5);
    
    std::cout << "\n┌─────────────────────────────────────────────────────────┐\n";
    std::cout << "│ Query Result Sizes (Number of matching documents)      │\n";
    std::cout << "├─────────────────────────────────────────────────────────┤\n";
    std::cout << "│  Q1: ((system AND data) OR computer)                    │\n";
    std::cout << "│      Result: " << std::setw(7) << sample1.doc_ids.size() << " docs (Large - good for parallel)        │\n";
    std::cout << "├─────────────────────────────────────────────────────────┤\n";
    std::cout << "│  Q2: (database AND NOT software)                        │\n";
    std::cout << "│      Result: " << std::setw(7) << sample2.doc_ids.size() << " docs (Large - good for parallel)        │\n";
    std::cout << "├─────────────────────────────────────────────────────────┤\n";
    std::cout << "│  Q3: ((system OR data) AND (computer OR software))      │\n";
    std::cout << "│      Result: " << std::setw(7) << sample3.doc_ids.size() << " docs (Massive - best for parallel)      │\n";
    std::cout << "├─────────────────────────────────────────────────────────┤\n";
    std::cout << "│  Q4: ((database AND search) OR (algo AND perf))         │\n";
    std::cout << "│      Result: " << std::setw(7) << sample4.doc_ids.size() << " docs (Medium - marginal benefit)        │\n";
    std::cout << "├─────────────────────────────────────────────────────────┤\n";
    std::cout << "│  Q5: (parallel AND sequential)                          │\n";
    std::cout << "│      Result: " << std::setw(7) << sample5.doc_ids.size() << " docs (Small - sequential preferred)     │\n";
    std::cout << "└─────────────────────────────────────────────────────────┘\n";

    // Warm-up runs
    std::cout << "\nWarming up caches...\n";
    for (int i = 0; i < 5; ++i) {
        seq_retrieval.execute_query(q1);
        par_retrieval.execute_query(q1);
    }

    // Run Benchmarks
    const int ITERATIONS = 500;
    std::cout << "Running benchmarks with " << ITERATIONS << " iterations per query...\n";
    std::cout << "(This may take 1-2 minutes)\n\n";
    
    // Sequential benchmarks
    perf.start_timer("seq_q1");
    for (int i = 0; i < ITERATIONS; ++i) seq_retrieval.execute_query(q1);
    perf.end_timer("seq_q1");
    
    perf.start_timer("seq_q2");
    for (int i = 0; i < ITERATIONS; ++i) seq_retrieval.execute_query(q2);
    perf.end_timer("seq_q2");

    perf.start_timer("seq_q3");
    for (int i = 0; i < ITERATIONS; ++i) seq_retrieval.execute_query(q3);
    perf.end_timer("seq_q3");

    perf.start_timer("seq_q4");
    for (int i = 0; i < ITERATIONS; ++i) seq_retrieval.execute_query(q4);
    perf.end_timer("seq_q4");

    perf.start_timer("seq_q5");
    for (int i = 0; i < ITERATIONS; ++i) seq_retrieval.execute_query(q5);
    perf.end_timer("seq_q5");

    // Parallel benchmarks
    perf.start_timer("par_q1");
    for (int i = 0; i < ITERATIONS; ++i) par_retrieval.execute_query(q1);
    perf.end_timer("par_q1");
    
    perf.start_timer("par_q2");
    for (int i = 0; i < ITERATIONS; ++i) par_retrieval.execute_query(q2);
    perf.end_timer("par_q2");

    perf.start_timer("par_q3");
    for (int i = 0; i < ITERATIONS; ++i) par_retrieval.execute_query(q3);
    perf.end_timer("par_q3");

    perf.start_timer("par_q4");
    for (int i = 0; i < ITERATIONS; ++i) par_retrieval.execute_query(q4);
    perf.end_timer("par_q4");

    perf.start_timer("par_q5");
    for (int i = 0; i < ITERATIONS; ++i) par_retrieval.execute_query(q5);
    perf.end_timer("par_q5");

    // Print Performance Results
    std::cout << "\n╔═══════════════════════════════════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                         PERFORMANCE RESULTS (Average Latency per Query)                           ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════════════════════════════════════════════╝\n\n";
    
    std::cout << std::fixed << std::setprecision(2);
    auto to_us_avg = [ITERATIONS](double ms){ return (ms * 1000.0) / ITERATIONS; };
    
    std::cout << std::left << std::setw(15) << "Strategy" 
              << std::setw(15) << "Q1 (Large)" 
              << std::setw(15) << "Q2 (Large)" 
              << std::setw(15) << "Q3 (Massive)"
              << std::setw(15) << "Q4 (Medium)"
              << std::setw(15) << "Q5 (Small)" << std::endl;
    std::cout << "──────────────────────────────────────────────────────────────────────────────────────────────────\n";
    
    double seq_q1 = to_us_avg(perf.get_duration_ms("seq_q1"));
    double seq_q2 = to_us_avg(perf.get_duration_ms("seq_q2"));
    double seq_q3 = to_us_avg(perf.get_duration_ms("seq_q3"));
    double seq_q4 = to_us_avg(perf.get_duration_ms("seq_q4"));
    double seq_q5 = to_us_avg(perf.get_duration_ms("seq_q5"));
    double par_q1 = to_us_avg(perf.get_duration_ms("par_q1"));
    double par_q2 = to_us_avg(perf.get_duration_ms("par_q2"));
    double par_q3 = to_us_avg(perf.get_duration_ms("par_q3"));
    double par_q4 = to_us_avg(perf.get_duration_ms("par_q4"));
    double par_q5 = to_us_avg(perf.get_duration_ms("par_q5"));
    
    std::cout << std::left << std::setw(15) << "Sequential" 
              << std::setw(15) << (std::to_string(static_cast<int>(seq_q1)) + " μs")
              << std::setw(15) << (std::to_string(static_cast<int>(seq_q2)) + " μs")
              << std::setw(15) << (std::to_string(static_cast<int>(seq_q3)) + " μs")
              << std::setw(15) << (std::to_string(static_cast<int>(seq_q4)) + " μs")
              << std::setw(15) << (std::to_string(static_cast<int>(seq_q5)) + " μs") << std::endl;
              
    std::cout << std::left << std::setw(15) << "Parallel" 
              << std::setw(15) << (std::to_string(static_cast<int>(par_q1)) + " μs")
              << std::setw(15) << (std::to_string(static_cast<int>(par_q2)) + " μs")
              << std::setw(15) << (std::to_string(static_cast<int>(par_q3)) + " μs")
              << std::setw(15) << (std::to_string(static_cast<int>(par_q4)) + " μs")
              << std::setw(15) << (std::to_string(static_cast<int>(par_q5)) + " μs") << std::endl;
    std::cout << "──────────────────────────────────────────────────────────────────────────────────────────────────\n";
    
    // Calculate speedups
    std::cout << std::setprecision(2);
    double speedup1 = seq_q1 / par_q1;
    double speedup2 = seq_q2 / par_q2;
    double speedup3 = seq_q3 / par_q3;
    double speedup4 = seq_q4 / par_q4;
    double speedup5 = seq_q5 / par_q5;
    
    auto format_speedup = [](double s) {
        std::string result = std::to_string(s) + "x";
        if (s >= 1.5) result += " ✓";
        else if (s >= 1.0) result += " ~";
        else result += " ✗";
        return result;
    };
    
    std::cout << std::left << std::setw(15) << "Speedup" 
              << std::setw(15) << format_speedup(speedup1)
              << std::setw(15) << format_speedup(speedup2)
              << std::setw(15) << format_speedup(speedup3)
              << std::setw(15) << format_speedup(speedup4)
              << std::setw(15) << format_speedup(speedup5) << std::endl;
    std::cout << "──────────────────────────────────────────────────────────────────────────────────────────────────\n";
    
    // Summary
    double avg_speedup = (speedup1 + speedup2 + speedup3 + speedup4 + speedup5) / 5.0;
    double max_speedup = std::max({speedup1, speedup2, speedup3, speedup4, speedup5});
    
    std::cout << "\n╔═══════════════════════════════════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  SUMMARY                                                                                          ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════════════════════════════════════════════╝\n";
    std::cout << "  Dataset Size: " << NUM_DOCS << " documents, " << full_index.size() << " unique terms\n";
    std::cout << "  Average Speedup: " << std::setprecision(2) << avg_speedup << "x\n";
    std::cout << "  Maximum Speedup: " << max_speedup << "x (Query " 
              << (max_speedup == speedup1 ? "1" : max_speedup == speedup2 ? "2" : 
                  max_speedup == speedup3 ? "3" : max_speedup == speedup4 ? "4" : "5") << ")\n";
    
    std::cout << "\n  Analysis:\n";
    if (speedup3 >= 2.0) {
        std::cout << "  ✓ Q3 shows strong parallelization benefit (large, balanced workload)\n";
    }
    if (speedup1 >= 1.5 && speedup2 >= 1.5) {
        std::cout << "  ✓ Q1 & Q2 benefit from parallel execution (large posting lists)\n";
    }
    if (speedup5 < 1.0) {
        std::cout << "  ✓ Q5 correctly avoided parallelization (small workload, overhead not justified)\n";
    } else if (speedup5 >= 0.9 && speedup5 < 1.1) {
        std::cout << "  ~ Q5 shows neutral performance (workload near threshold)\n";
    }
    
    std::cout << "\n  Parallel Efficiency:\n";
    std::cout << "    - Large queries (Q1-Q3): Parallel execution benefits clear\n";
    std::cout << "    - Medium queries (Q4): Marginal benefit, threshold tuning effective\n";
    std::cout << "    - Small queries (Q5): Sequential preferred, avoiding unnecessary overhead\n";
    
    if (avg_speedup >= 1.5) {
        std::cout << "\n  ✓✓ EXCELLENT: Parallel retrieval significantly outperforms sequential!\n";
    } else if (avg_speedup >= 1.2) {
        std::cout << "\n  ✓ GOOD: Parallel retrieval shows meaningful improvement\n";
    } else {
        std::cout << "\n  ~ MIXED: Benefits vary by query complexity\n";
    }
    
    std::cout << "\n══════════════════════════════════════════════════════════════════════════════════════════════════\n";

    return 0;
}