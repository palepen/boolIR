#ifndef EVALUATOR_H
#define EVALUATOR_H

#include <unordered_map>
#include <unordered_set>
#include <string>
#include <vector>
#include "common/utils.h"

using Qrels = std::unordered_map<std::string, std::unordered_set<unsigned int>>;

// Evaluation metrics
struct EvaluationResults
{
    double precision_at_10;
    double mean_average_precision;
    double mean_reciprocal_rank;
    double ndcg_at_10;
    double dcg_at_10;
};

class Evaluator
{
private:
    const Qrels &qrels_;

    double calculate_precision_at_k(
        const std::vector<SearchResult> &results,
        const std::string &query_id,
        size_t k) const;

    double calculate_average_precision(
        const std::vector<SearchResult> &results,
        const std::string &query_id) const;

    double calculate_dcg_at_k(
        const std::vector<SearchResult> &results,
        const std::string &query_id,
        size_t k) const;

    double calculate_idcg_at_k(
        const std::string &query_id,
        size_t k) const;

public:
    explicit Evaluator(const Qrels &qrels);

    EvaluationResults evaluate(
        const std::unordered_map<std::string, std::vector<SearchResult>> &all_results) const;
};

#endif