#ifndef EVALUATION_H
#define EVALUATION_H

#include "retrieval.h"

typedef struct EvaluationMetrics {
    double precision_at_k[10];  // P@1, P@5, P@10, etc.
    double recall_at_k[10];
    double map_score;
    double ndcg_score;
    double mrr_score;
} EvaluationMetrics;

// Function declarations
EvaluationMetrics* evaluate_results(ResultSet *results, const std::unordered_set<std::string>& relevant_docs);
void print_metrics(EvaluationMetrics *metrics);
void save_metrics_to_file(EvaluationMetrics *metrics, const char *filename);

#endif