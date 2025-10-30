#ifndef SYSTEM_CONTROLLER_H
#define SYSTEM_CONTROLLER_H

#include "retrieval/retriever.h" 
#include "retrieval/query_expander.h"
#include "retrieval/query_preprocessor.h"
#include "document_store.h"
#include "common/utils.h"
#include <memory>
#include <vector>
#include <string>

class HighPerformanceIRSystem
{
public:
    HighPerformanceIRSystem(
        const std::string &index_path,
        const std::string &synonym_path,
        size_t num_shards = 64 
    );

    std::vector<SearchResult> search_boolean(
        const std::string &query_str, bool print_log);

private:
    std::unique_ptr<QueryNode> expand_query(const std::string &query_str);

    std::unique_ptr<QueryExpander> query_expander_;
    std::unique_ptr<QueryPreprocessor> query_preprocessor_;
    std::unique_ptr<ParallelRetriever> retriever_; 
};

#endif