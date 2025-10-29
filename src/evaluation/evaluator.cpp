#include "evaluation/evaluator.h"
#include <numeric>
#include <algorithm>
#include <cmath>

Evaluator::Evaluator(const Qrels &qrels) : qrels_(qrels) {}

double Evaluator::calculate_precision_at_k(
    const std::vector<SearchResult> &results,
    const std::string &query_id,
    size_t k) const
{
    auto it = qrels_.find(query_id);
    if (it == qrels_.end() || it->second.empty())
    {
        return 0.0;
    }
    const auto &relevant_docs = it->second;

    size_t relevant_found = 0;
    size_t limit = std::min(k, results.size());

    for (size_t i = 0; i < limit; ++i)
    {
        if (relevant_docs.count(results[i].doc_id))
        {
            relevant_found++;
        }
    }

    return (k > 0) ? static_cast<double>(relevant_found) / k : 0.0;
}

double Evaluator::calculate_average_precision(
    const std::vector<SearchResult> &results,
    const std::string &query_id) const
{
    auto it = qrels_.find(query_id);
    if (it == qrels_.end() || it->second.empty())
    {
        return 0.0;
    }
    const auto &relevant_docs = it->second;

    double sum_of_precisions = 0.0;
    size_t relevant_found = 0;

    for (size_t i = 0; i < results.size(); ++i)
    {
        if (relevant_docs.count(results[i].doc_id))
        {
            relevant_found++;
            sum_of_precisions += static_cast<double>(relevant_found) / (i + 1);
        }
    }

    return (relevant_found > 0) ? sum_of_precisions / relevant_docs.size() : 0.0;
}

double Evaluator::calculate_dcg_at_k(
    const std::vector<SearchResult> &results,
    const std::string &query_id,
    size_t k) const
{
    auto it = qrels_.find(query_id);
    if (it == qrels_.end() || it->second.empty())
    {
        return 0.0;
    }
    const auto &relevant_docs = it->second;

    double dcg = 0.0;
    size_t limit = std::min(k, results.size());

    for (size_t i = 0; i < limit; ++i)
    {
        // Gain = 1 if relevant, 0 otherwise
        double gain = relevant_docs.count(results[i].doc_id) ? 1.0 : 0.0;

        // DCG formula: gain / log2(position + 1)
        // Position is 1-indexed for this formula
        double discount = std::log2(static_cast<double>(i + 2));
        dcg += gain / discount;
    }

    return dcg;
}

double Evaluator::calculate_idcg_at_k(
    const std::string &query_id,
    size_t k) const
{
    auto it = qrels_.find(query_id);
    if (it == qrels_.end() || it->second.empty())
    {
        return 0.0;
    }

    // Ideal DCG: all relevant docs at top positions
    size_t num_relevant = it->second.size();
    size_t limit = std::min(k, num_relevant);

    double idcg = 0.0;
    for (size_t i = 0; i < limit; ++i)
    {
        // All positions have gain = 1 in ideal case
        double discount = std::log2(static_cast<double>(i + 2));
        idcg += 1.0 / discount;
    }

    return idcg;
}

EvaluationResults Evaluator::evaluate(
    const std::unordered_map<std::string, std::vector<SearchResult>> &all_results) const
{
    EvaluationResults final_metrics;
    if (all_results.empty())
    {
        return final_metrics;
    }

    double total_ap = 0.0;
    double total_rr = 0.0;
    double total_p10 = 0.0;
    double total_dcg10 = 0.0;
    double total_ndcg10 = 0.0;

    for (const auto &pair : all_results)
    {
        const std::string &query_id = pair.first;
        const auto &results = pair.second;

        // MAP
        total_ap += calculate_average_precision(results, query_id);

        // P@10
        total_p10 += calculate_precision_at_k(results, query_id, 10);

        // MRR
        auto it = qrels_.find(query_id);
        if (it != qrels_.end())
        {
            const auto &relevant_docs = it->second;
            for (size_t i = 0; i < results.size(); ++i)
            {
                if (relevant_docs.count(results[i].doc_id))
                {
                    total_rr += 1.0 / (i + 1.0);
                    break;
                }
            }
        }

        // DCG@10 and NDCG@10
        double dcg = calculate_dcg_at_k(results, query_id, 10);
        double idcg = calculate_idcg_at_k(query_id, 10);

        total_dcg10 += dcg;

        // NDCG = DCG / IDCG (avoid division by zero)
        if (idcg > 0.0)
        {
            total_ndcg10 += dcg / idcg;
        }
    }

    size_t num_queries = all_results.size();
    final_metrics.mean_average_precision = total_ap / num_queries;
    final_metrics.mean_reciprocal_rank = total_rr / num_queries;
    final_metrics.precision_at_10 = total_p10 / num_queries;
    final_metrics.dcg_at_10 = total_dcg10 / num_queries;
    final_metrics.ndcg_at_10 = total_ndcg10 / num_queries;

    return final_metrics;
}