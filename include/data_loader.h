#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include "indexing/document.h"
#include "evaluation/evaluator.h"
#include <string>
#include <unordered_map>
#include <filesystem>

namespace fs = std::filesystem;

using DocNameToIdMap = std::unordered_map<std::string, unsigned int>;

using IdToDocNameMap = std::unordered_map<unsigned int, std::string>;

struct DocumentLoadResult
{
    DocumentCollection documents;
    DocNameToIdMap doc_name_to_id; // "ug7v899j" -> 0
    IdToDocNameMap id_to_doc_name; // 0 -> "ug7v899j"
};

// Updated function declarations
DocumentLoadResult load_trec_documents(const std::string &corpus_dir);
Qrels load_trec_qrels(const std::string &qrels_path, const DocNameToIdMap &doc_name_to_id);
std::unordered_map<std::string, std::string> load_trec_topics(const std::string &topics_path);

#endif // DATA_LOADER_H