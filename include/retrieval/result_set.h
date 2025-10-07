#ifndef RESULT_SET_H
#define RESULT_SET_H

#include <vector>
#include <iostream>
#include <algorithm>

class ResultSet {
    public:
        std::vector<u_int> doc_ids;
        static ResultSet intersect_sets(const ResultSet& a, const ResultSet& b);
        static ResultSet union_sets(const ResultSet& a, const ResultSet& b);
        static ResultSet differ_sets(const ResultSet& a, const ResultSet& b);
        
        void print() const {
            for (u_int id : doc_ids) {
                std:: cout << id << " ";
            }
            std::cout << "Total: " << doc_ids.size() << ")" << std::endl;
        }

};

#endif