#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <sstream>
#include <numeric>
#include <cmath>
#include "evaluation.h"

// Helper function to parse the ground truth file (qrels)
// Assumes TREC format: "query_id 0 doc_id relevance"
// We only care about entries where relevance is > 0.
static std::unordered_map<int, std::unordered_set<std::string>> load_ground_truth(const char* ground_truth_file) {
    std::unordered_map<int, std::unordered_set<std::string>> ground_truth;
    std::ifstream file(ground_truth_file);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open ground truth file: " << ground_truth_file << std::endl;
        return ground_truth;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        int query_id, relevance;
        std::string doc_id, unused;
        ss >> query_id >> unused >> doc_id >> relevance;
        if (relevance > 0) {
            ground_truth[query_id].insert(doc_id);
        }
    }
    return ground_truth;
}

// Main evaluation function
EvaluationMetrics* evaluate_results(ResultSet* results, const std::unordered_set<std::string>& relevant_docs) {
    auto* metrics = new EvaluationMetrics();
    if (relevant_docs.empty()) {
        return metrics; // No relevant docs for this query, all metrics are 0
    }

    int relevant_found = 0;
    double average_precision = 0.0;
    int total_relevant = relevant_docs.size();

    for (int k = 0; k < results->num_results; ++k) {
        // NOTE: This assumes doc_id can be converted to a comparable string.
        // You may need to map your internal numeric doc_id back to its original filename/string identifier.
        // For this example, we'll simulate this with std::to_string.
        std::string current_doc_id = std::to_string(results->results[k].doc_id);

        if (relevant_docs.count(current_doc_id)) {
            relevant_found++;
            average_precision += (double)relevant_found / (k + 1.0);
        }

        // Calculate P@K and R@K for k+1
        if (k + 1 == 1 || k + 1 == 5 || k + 1 == 10) {
            int index = -1;
            if (k + 1 == 1) index = 0;
            else if (k + 1 == 5) index = 1;
            else if (k + 1 == 10) index = 2; // Assuming P@1, P@5, P@10
            
            if(index != -1) {
                metrics->precision_at_k[index] = (double)relevant_found / (k + 1.0);
                metrics->recall_at_k[index] = (double)relevant_found / total_relevant;
            }
        }
    }

    metrics->map_score = average_precision / total_relevant;
    // NDCG and MRR are left as an exercise.
    metrics->ndcg_score = 0.0;
    metrics->mrr_score = 0.0;

    return metrics;
}

void print_metrics(EvaluationMetrics* metrics) {
    if (!metrics) return;
    printf("  Precision@1:  %.4f\n", metrics->precision_at_k[0]);
    printf("  Precision@5:  %.4f\n", metrics->precision_at_k[1]);
    printf("  Precision@10: %.4f\n", metrics->precision_at_k[2]);
    printf("  Recall@1:     %.4f\n", metrics->recall_at_k[0]);
    printf("  Recall@5:     %.4f\n", metrics->recall_at_k[1]);
    printf("  Recall@10:    %.4f\n", metrics->recall_at_k[2]);
    printf("  MAP Score:    %.4f\n", metrics->map_score);
}

// This is a placeholder for saving; implementation is up to you.
void save_metrics_to_file(EvaluationMetrics *metrics, const char *filename) {
    // Implementation to write metrics to a CSV or JSON file.
    (void)metrics; // unused
    (void)filename; // unused
    std::cout << "Saving metrics to file is not yet implemented." << std::endl;
}