#include "indexing/posting_list.h"
#include <iostream>
#include <vector>

void print_postings(const PostingList& pl) {
    for (int id : pl.get_postings()) {
        std::cout << id << " ";
    }
    std::cout << std::endl;
}

int main() {
    std::cout << "--- Testing PostingsList ---" << std::endl;
    PostingList pl;

    std::cout << "Adding unsorted numbers with duplicates: 50, 10, 80, 10, 90, 50" << std::endl;
    pl.add_document(50);
    pl.add_document(10);
    pl.add_document(80);
    pl.add_document(10); // Duplicate
    pl.add_document(90);
    pl.add_document(50); // Duplicate

    std::cout << "Final list: ";
    print_postings(pl);

    std::cout << "Expected list: 10 50 80 90" << std::endl;

    // A simple check for correctness
    const std::vector<int> expected = {10, 50, 80, 90};
    if (pl.get_postings() == expected) {
        std::cout << "\nTest Passed: The list is sorted and unique." << std::endl;
    } else {
        std::cout << "\nTest Failed: The list is incorrect." << std::endl;
    }
    
    return 0;
}