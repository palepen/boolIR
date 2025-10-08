#include "system_controller.h"
#include <iostream>
#include <iomanip>
#include <thread>

// Function to generate a sample collection of documents
DocumentCollection create_sample_documents(size_t num_docs) {
    DocumentCollection docs;
    docs.reserve(num_docs);
    for (size_t i = 0; i < num_docs; ++i) {
        std::string content = "The quick brown fox jumps over the lazy dog. ";
        if (i % 5 == 0) content += "apple banana ";
        if (i % 7 == 0) content += "apollo space program ";
        if (i % 11 == 0) content += "machine learning ";
        docs.push_back({static_cast<unsigned int>(i), content});
    }
    return docs;
}

int main() {
    const char* model_path = "models/bert_model.onnx";
    const size_t num_cores = std::thread::hardware_concurrency();
    const size_t num_docs = 5000;

    std::cout << "--- Stage 4: End-to-End System Benchmark ---" << std::endl;

    // 1. Initialize the system
    HighPerformanceIRSystem system(num_cores, model_path);

    // 2. Create sample data and build the index
    DocumentCollection documents = create_sample_documents(num_docs);
    system.build_index(documents);

    // 3. Define a query and run the full search pipeline
    std::string query = "apollo";
    std::vector<SearchResult> results = system.search(query);

    // 4. Print the top 10 results
    std::cout << "\n--- Top 10 Search Results for Query: '" << query << "' ---" << std::endl;
    std::cout << std::left << std::setw(10) << "Rank"
              << std::setw(15) << "Document ID"
              << std::setw(15) << "Score" << std::endl;
    std::cout << std::string(40, '-') << std::endl;

    for (size_t i = 0; i < std::min((size_t)10, results.size()); ++i) {
        std::cout << std::left << std::setw(10) << i + 1
                  << std::setw(15) << results[i].doc_id
                  << std::fixed << std::setprecision(4) << std::setw(15) << results[i].score
                  << std::endl;
    }
    
    std::cout << "------------------------------------------" << std::endl;

    return 0;
}