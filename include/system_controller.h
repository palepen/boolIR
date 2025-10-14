#ifndef SYSTEM_CONTROLLER_H
#define SYSTEM_CONTROLLER_H

#include "retrieval/optimized_parallel_retrieval.h"
#include "retrieval/query_expander.h"
#include "retrieval/query_preprocessor.h"
#include "retrieval/pre_ranker.h"
#include "indexing/document.h"
#include "common_types.h"
#include <memory>
#include <vector>
#include <string>

class HighPerformanceIRSystem {
public:
    HighPerformanceIRSystem(
        const std::string& index_path, 
        const std::string& synonym_path,
        std::unique_ptr<PreRanker> pre_ranker
    );
    
    // Returns a pre-ranked list of candidate documents from Boolean retrieval
    std::vector<SearchResult> search_boolean(
        const std::string& query_str,
        const DocumentCollection& documents
    );
    
private:
    std::unique_ptr<QueryNode> expand_query(const std::string& query_str);

    std::unique_ptr<QueryExpander> query_expander_;
    std::unique_ptr<QueryPreprocessor> query_preprocessor_;
    std::unique_ptr<OptimizedParallelRetrieval> retriever_;
    std::unique_ptr<PreRanker> pre_ranker_;
};

#endif