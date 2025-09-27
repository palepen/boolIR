#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include "ui.h"
#include "indexing.h"
#include "retrieval.h"
#include "neural_ranking.h"

// Function to display how to use the program
void show_usage(const char* program_name) {
    std::cerr << "Usage: " << program_name << " --mode <interactive|baseline|neural> --dataset <path_to_docs> [options]" << std::endl;
}

// Function to display search results
void display_results(ResultSet* results) {
    if (!results || results->num_results == 0) {
        std::cout << "  > No results found." << std::endl;
        return;
    }
    std::cout << "  > Found " << results->num_results << " documents." << std::endl;
    // Optional: Print top N doc IDs
    int count = 0;
    for (int i = 0; i < results->num_results && count < 10; ++i, ++count) {
        std::cout << "    - Doc ID: " << results->results[i].doc_id << std::endl;
    }
}

// Main function for the interactive mode
void run_interactive_mode(InvertedIndex* index, NeuralRanker* /*ranker*/) { // Silence unused parameter warning
    std::string query_str;
    std::cout << "\nEntering interactive mode. Type 'EXIT' to quit." << std::endl;
    std::cout << "Example query: 'retrieval AND model OR parallel AND search'" << std::endl;

    while (true) {
        std::cout << "\n> Enter your query: ";
        std::getline(std::cin, query_str);

        if (query_str == "EXIT") break;
        if (query_str.empty()) continue;

        // --- 1. Run Sequential Search ---
        BooleanQuery* seq_query = parse_boolean_query(query_str.c_str());
        auto start_seq = std::chrono::high_resolution_clock::now();
        ResultSet* results_seq = execute_sequential_search(index, seq_query);
        auto end_seq = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed_seq = end_seq - start_seq;
        
        std::cout << "\n--- Sequential Search ---" << std::endl;
        display_results(results_seq);
        std::cout << "  > Sequential search took: " << elapsed_seq.count() << " ms." << std::endl;

        // --- 2. Run Parallel Search ---
        QueryNode* par_query = parse_query_to_tree(query_str);
        auto start_par = std::chrono::high_resolution_clock::now();
        ResultSet* results_par = execute_parallel_search(index, par_query);
        auto end_par = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed_par = end_par - start_par;

        std::cout << "\n--- Parallel Search ---" << std::endl;
        display_results(results_par);
        std::cout << "  > Parallel search took:   " << elapsed_par.count() << " ms." << std::endl;

        // --- 3. Calculate and Display Speedup ---
        if (elapsed_par.count() > 0 && elapsed_seq.count() > 0) {
            double speedup = elapsed_seq.count() / elapsed_par.count();
            printf("\n  Speedup: %.2fx\n", speedup);
        }

        // --- 4. Cleanup ---
        free_boolean_query(seq_query);
        free_result_set(results_seq);
        free_query_tree(par_query);
        free_result_set(results_par);
    }
}

// Baseline mode for batch processing
void run_baseline_mode(InvertedIndex *index, int argc, char *argv[]) {
    std::string query_str;
    for (int i = 3; i < argc; ++i) {
        if (std::string(argv[i]) == "--query" && i + 1 < argc) {
            query_str = argv[i+1];
            break;
        }
    }

    if (query_str.empty()) {
        std::cerr << "No query provided for baseline mode." << std::endl;
        show_usage(argv[0]);
        return;
    }

    std::cout << "Running baseline search for query: '" << query_str << "'" << std::endl;
    BooleanQuery* query = parse_boolean_query(query_str.c_str());
    
    // ** THE FIX IS HERE **
    // Changed boolean_search to execute_sequential_search
    ResultSet* results = execute_sequential_search(index, query);
    
    display_results(results);
    free_boolean_query(query);
    free_result_set(results);
}


// Neural mode for batch processing (placeholder)
void run_neural_mode(InvertedIndex* /*index*/, NeuralRanker* ranker, int /*argc*/, char** /*argv*/) {
     if (!ranker) {
        std::cerr << "Neural ranker not initialized." << std::endl;
        return;
    }
    std::cout << "Neural mode is not fully implemented yet." << std::endl;
}