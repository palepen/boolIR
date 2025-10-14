#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include "indexing/document.h"    // Assumed to define Document and DocumentCollection
#include "evaluation/evaluator.h" // Assumed to define Qrels
#include <string>
#include <unordered_map>
#include <filesystem>

namespace fs = std::filesystem;

// Type alias for mapping document names to internal IDs
using DocNameToIdMap = std::unordered_map<std::string, unsigned int>;

// Function declarations for loading TREC-COVID data
std::pair<DocumentCollection, DocNameToIdMap> load_trec_documents(const std::string &corpus_dir);
Qrels load_trec_qrels(const std::string &qrels_path, const DocNameToIdMap &doc_name_to_id);
std::unordered_map<std::string, std::string> load_trec_topics(const std::string &topics_path);

#endif // DATA_LOADER_H