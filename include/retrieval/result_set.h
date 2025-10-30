#ifndef RESULT_SET_H
#define RESULT_SET_H

#include <vector>
#include <iostream>
#include <algorithm>

/**
 * Pure Boolean ResultSet
 * Contains only document IDs without scores
 */
class ResultSet {
public:
    std::vector<unsigned int> doc_ids;
    
    static ResultSet intersect_sets(const ResultSet& a, const ResultSet& b);
    static ResultSet union_sets(const ResultSet& a, const ResultSet& b);
    static ResultSet differ_sets(const ResultSet& a, const ResultSet& b);
    
    void print() const {
        for (size_t i = 0; i < std::min(size_t(10), doc_ids.size()); ++i) {
            std::cout << doc_ids[i] << " ";
        }
        std::cout << "\n(Total: " << doc_ids.size() << " documents)" << std::endl;
    }
};

#endif