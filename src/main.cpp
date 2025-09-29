// #include <iostream>
// #include <vector>
// #include "indexing.h"

// int main() {
//     // Minimal test dataset
//     std::vector<Document> docs = {
//         {1, "The quick brown fox jumps over the lazy dog"},
//         {2, "A fast brown fox leaps over a sleepy dog"},
//         {3, "Information retrieval is fascinating"},
//         {4, "Boolean retrieval systems are fast but not ranked"},
//         {5, "Neural re ranking improves relevance"}
//     };

//     std::cout << "=== Testing Parallel Indexing ===\n";
    
//     InvertedIndex index;
//     index.build(docs);

//     // Test queries
//     std::vector<std::string> test_terms = {"fox", "retrieval", "fast", "neural"};
    
//     for (const auto& term : test_terms) {
//         auto postings = index.get_postings(term);
//         std::cout << "Postings for '" << term << "': ";
//         for (int id : postings) {
//             std::cout << id << " ";
//         }
//         std::cout << "\n";
//     }

//     std::cout << "\n✅ Indexing test completed successfully!\n";
//     return 0;
// }