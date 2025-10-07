#include "retrieval/result_set.h"

ResultSet ResultSet::intersect_sets(const ResultSet &a, const ResultSet &b)
{
    ResultSet result;
    const auto &a_ids = a.doc_ids;
    const auto &b_ids = b.doc_ids;

    // Early exit optimizations
    if (a_ids.empty() || b_ids.empty())
    {
        return result;
    }

    // Reserve space to avoid reallocations
    result.doc_ids.reserve(std::min(a_ids.size(), b_ids.size()));

    size_t i = 0, j = 0;

    // Galloping/exponential search for large size differences
    if (a_ids.size() > b_ids.size() * 10 || b_ids.size() > a_ids.size() * 10)
    {
        // Use binary search when one list is much larger
        const auto &smaller = (a_ids.size() < b_ids.size()) ? a_ids : b_ids;
        const auto &larger = (a_ids.size() < b_ids.size()) ? b_ids : a_ids;

        for (u_int val : smaller)
        {
            if (std::binary_search(larger.begin(), larger.end(), val))
            {
                result.doc_ids.push_back(val);
            }
        }
        return result;
    }

    // Standard two-pointer merge for similar-sized lists
    while (i < a_ids.size() && j < b_ids.size())
    {
        if (a_ids[i] < b_ids[j])
        {
            i++;
        }
        else if (b_ids[j] < a_ids[i])
        {
            j++;
        }
        else
        {
            result.doc_ids.push_back(a_ids[i]);
            i++;
            j++;
        }
    }
    return result;
}

ResultSet ResultSet::union_sets(const ResultSet &a, const ResultSet &b)
{
    ResultSet result;
    const auto &a_ids = a.doc_ids;
    const auto &b_ids = b.doc_ids;

    // Early exit optimizations
    if (a_ids.empty())
        return b;
    if (b_ids.empty())
        return a;

    // Reserve space to avoid reallocations
    result.doc_ids.reserve(a_ids.size() + b_ids.size());

    size_t i = 0, j = 0;
    while (i < a_ids.size() || j < b_ids.size())
    {
        if (i < a_ids.size() && (j == b_ids.size() || a_ids[i] < b_ids[j]))
        {
            result.doc_ids.push_back(a_ids[i]);
            i++;
        }
        else if (j < b_ids.size() && (i == a_ids.size() || b_ids[j] < a_ids[i]))
        {
            result.doc_ids.push_back(b_ids[j]);
            j++;
        }
        else if (i < a_ids.size() && j < b_ids.size())
        {
            result.doc_ids.push_back(a_ids[i]);
            i++;
            j++;
        }
    }
    return result;
}

ResultSet ResultSet::differ_sets(const ResultSet &a, const ResultSet &b)
{
    ResultSet result;
    const auto &a_ids = a.doc_ids;
    const auto &b_ids = b.doc_ids;

    // Early exit optimizations
    if (a_ids.empty())
        return result;
    if (b_ids.empty())
        return a;

    // Reserve space
    result.doc_ids.reserve(a_ids.size());

    size_t i = 0, j = 0;

    // Optimized difference operation
    while (i < a_ids.size())
    {
        if (j == b_ids.size() || a_ids[i] < b_ids[j])
        {
            result.doc_ids.push_back(a_ids[i]);
            i++;
        }
        else if (b_ids[j] < a_ids[i])
        {
            j++;
        }
        else
        {
            // Skip elements in both sets
            i++;
            j++;
        }
    }
    return result;
}