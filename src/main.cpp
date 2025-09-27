#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <cilk/cilk_api.h>
#include <fstream>
#include <sstream>
#include <numeric>
#include <unordered_map>
#include <unordered_set>

#include "indexing.h"
#include "retrieval.h"
#include "neural_ranking.h"
#include "ui.h"
#include "evaluation.h" // Include the evaluation header

// --- Helper Functions for Evaluation Mode ---

/**
 * @brief Loads and parses queries directly from a TREC-formatted topic file.
 * This function reads a file with multi-line topic blocks (e.g., from NIST)
 * and extracts the topic number and title.
 * @param query_file Path to the TREC topic file.
 * @return A vector of pairs, where each pair contains a query ID and its text.
 */
std::vector<std::pair<int, std::string>> load_queries(const std::string& query_file) {
    std::vector<std::pair<int, std::string>> queries;
    std::ifstream file(query_file);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open query file: " << query_file << std::endl;
        return queries;
    }

    std::string line;
    int current_id = -1;
    std::string current_title;
    bool in_title = false;

    while (std::getline(file, line)) {
        // Find the topic number
        if (line.rfind("<num> Number:", 0) == 0) {
            size_t last_space = line.rfind(' ');
            // Ensure a space was found and there is content after it
            if (last_space != std::string::npos && last_space + 1 < line.length()) {
                std::string num_str = line.substr(last_space + 1);
                try {
                    // Safely attempt to convert the string to an integer
                    current_id = std::stoi(num_str);
                } catch (const std::exception& e) {
                    std::cerr << "Warning: Could not parse topic number. Line: \"" << line << "\". Error: " << e.what() << std::endl;
                    current_id = -1; // Reset on failure
                }
            }
        }
        // Find the start of the title
        else if (line.rfind("<title>", 0) == 0) {
            current_title = line.substr(7);
            size_t first_char = current_title.find_first_not_of(" \t");
            if (first_char != std::string::npos) {
                current_title = current_title.substr(first_char);
            }
            if (current_title.empty()) {
                in_title = true;
            }
        }
        // Handle cases where the title is on the line after the tag
        else if (in_title) {
            current_title = line;
            size_t first_char = current_title.find_first_not_of(" \t");
            if (first_char != std::string::npos) {
                current_title = current_title.substr(first_char);
            }
            in_title = false;
        }
        // End of a topic block
        else if (line.rfind("</top>", 0) == 0) {
            if (current_id != -1 && !current_title.empty()) {
                queries.push_back({current_id, current_title});
            }
            // Reset for the next topic
            current_id = -1;
            current_title.clear();
            in_title = false;
        }
    }

    std::cout << "Loaded and parsed " << queries.size() << " queries." << std::endl;
    return queries;
}

/**
 * @brief Loads ground truth relevance judgments from a qrels file.
 * @param qrels_file Path to the qrels file (TREC format: "qid 0 docid rel").
 * @return An unordered_map where the key is the query ID and the value is a
 * set of relevant document IDs.
 */
std::unordered_map<int, std::unordered_set<std::string>> load_ground_truth(const std::string& qrels_file) {
    std::unordered_map<int, std::unordered_set<std::string>> ground_truth;
    std::ifstream file(qrels_file);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open qrels file: " << qrels_file << std::endl;
        return ground_truth;
    }
    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        int query_id, relevance;
        std::string doc_id_str, unused_zero;
        ss >> query_id >> unused_zero >> doc_id_str >> relevance;
        if (relevance > 0) {
            ground_truth[query_id].insert(doc_id_str);
        }
    }
    std::cout << "Loaded ground truth for " << ground_truth.size() << " queries." << std::endl;
    return ground_truth;
}


int main(int argc, char *argv[]) {
    std::cout << "Boolean Retrieval System with Neural Re-ranking (OpenCilk)" << std::endl;
    std::cout << "=========================================================" << std::endl;
    
    if (__cilkrts_get_nworkers() > 0) {
        std::cout << "OpenCilk Workers: " << __cilkrts_get_nworkers() << std::endl;
    } else {
        std::cout << "OpenCilk scheduler not running." << std::endl;
    }

    // --- Argument Parsing ---
    if (argc < 3) {
        show_usage(argv[0]);
        return 1;
    }
    
    std::string mode;
    std::string dataset_path;
    std::string query_file;
    std::string qrels_file;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--mode" && i + 1 < argc) {
            mode = argv[++i];
        } else if (arg == "--dataset" && i + 1 < argc) {
            dataset_path = argv[++i];
        } else if (arg == "--queries" && i + 1 < argc) {
            query_file = argv[++i];
        } else if (arg == "--qrels" && i + 1 < argc) {
            qrels_file = argv[++i];
        }
    }

    if (mode.empty() || dataset_path.empty()) {
        show_usage(argv[0]);
        return 1;
    }
    if (mode == "evaluation" && (query_file.empty() || qrels_file.empty())) {
        std::cerr << "Error: Evaluation mode requires --queries and --qrels arguments." << std::endl;
        show_usage(argv[0]);
        return 1;
    }

    // --- System Initialization ---
    auto start_indexing = std::chrono::high_resolution_clock::now();
    InvertedIndex* index = create_inverted_index();
    build_index_parallel(index, dataset_path.c_str());
    auto end_indexing = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> indexing_duration = end_indexing - start_indexing;
    std::cout << "\nIndex built in " << indexing_duration.count() << " seconds." << std::endl;
    
    NeuralRanker* ranker = nullptr;
    if (mode == "neural" || mode == "interactive") {
        ranker = initialize_neural_ranker();
    }

    // --- Mode Execution ---
    if (mode == "interactive") {
        run_interactive_mode(index, ranker);
    } else if (mode == "baseline") {
        run_baseline_mode(index, argc, argv);
    } else if (mode == "neural") {
        run_neural_mode(index, ranker, argc, argv);
    } else if (mode == "evaluation") {
        std::cout << "\n--- Running Evaluation Mode ---" << std::endl;
        
        auto queries = load_queries(query_file);
        auto ground_truth = load_ground_truth(qrels_file);

        if (queries.empty() || ground_truth.empty()) {
            std::cerr << "Cannot run evaluation due to errors loading files." << std::endl;
            return 1;
        }

        double total_map = 0.0;
        std::vector<double> total_p1, total_p5, total_p10;
        int evaluated_queries = 0;

        for (const auto& q_pair : queries) {
            int query_id = q_pair.first;
            const std::string& query_text = q_pair.second;

            if (ground_truth.find(query_id) == ground_truth.end()) {
                continue; // Skip queries with no relevance judgments
            }
            
            // Execute search
            QueryNode* query_tree = parse_query_to_tree(query_text);
            ResultSet* results = execute_parallel_search(index, query_tree);
            
            // Evaluate results
            const auto& relevant_docs = ground_truth.at(query_id);
            EvaluationMetrics* metrics = evaluate_results(results, relevant_docs);
            
            // Aggregate metrics
            total_map += metrics->map_score;
            total_p1.push_back(metrics->precision_at_k[0]);
            total_p5.push_back(metrics->precision_at_k[1]);
            total_p10.push_back(metrics->precision_at_k[2]);
            
            evaluated_queries++;

            // Cleanup for this query
            free_query_tree(query_tree);
            free_result_set(results);
            delete metrics;
        }

        if (evaluated_queries > 0) {
            std::cout << "\n--- Overall Evaluation Metrics ---" << std::endl;
            printf("Evaluated %d queries.\n", evaluated_queries);
            printf("Mean Average Precision (MAP): %.4f\n", total_map / evaluated_queries);
            printf("Mean Precision@1:           %.4f\n", std::accumulate(total_p1.begin(), total_p1.end(), 0.0) / evaluated_queries);
            printf("Mean Precision@5:           %.4f\n", std::accumulate(total_p5.begin(), total_p5.end(), 0.0) / evaluated_queries);
            printf("Mean Precision@10:          %.4f\n", std::accumulate(total_p10.begin(), total_p10.end(), 0.0) / evaluated_queries);
            std::cout << "------------------------------------" << std::endl;
        } else {
            std::cout << "\nNo queries were evaluated. Check if query IDs match in queries and qrels files." << std::endl;
        }

    } else {
        std::cerr << "Invalid mode specified." << std::endl;
        show_usage(argv[0]);
    }

    // --- Cleanup ---
    std::cout << "\nCleaning up resources..." << std::endl;
    free_inverted_index(index);
    if (ranker) {
        free_neural_ranker(ranker);
    }
    std::cout << "Shutdown complete." << std::endl;

    return 0;
}