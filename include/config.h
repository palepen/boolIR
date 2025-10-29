#ifndef CONFIG_H
#define CONFIG_H

#include <string>
#include <stddef.h>

/**
 * @brief Central configuration for the entire IR system.
 *
 * This namespace holds all file paths, indexing parameters,
 * and reranking hyperparameters in one place.
 */
namespace Config
{
    // --- File Paths ---
    const std::string CORPUS_DIR = "data/cord19-trec-covid_corpus";
    const std::string TOPICS_PATH = "data/topics.cord19-trec-covid.txt";
    const std::string QRELS_PATH = "data/qrels.cord19-trec-covid.txt";
    const std::string SYNONYM_PATH = "data/synonyms.txt";
    const std::string MODEL_PATH = "models/bert_model.pt";
    const std::string VOCAB_PATH = "models/vocab.txt";
    const std::string INDEX_PATH = "index";
    const std::string TEMP_PATH = "index/temp";
    const std::string RESULTS_CSV_PATH = "results/all_benchmarks.csv";
    const std::string PLOTS_DIR = "results/plots";

    // --- Indexing Parameters ---
    constexpr size_t DEFAULT_NUM_SHARDS = 64;
    constexpr size_t DEFAULT_BLOCK_SIZE_MB = 256;

    // --- Reranking Hyperparameters ---
    constexpr size_t MAX_RERANK_CANDIDATES = 1024;
    constexpr int64_t MAX_SEQ_LEN = 512;
    constexpr size_t DOCUMENT_TRUNCATE_WORDS = 256;
    constexpr size_t DEFAULT_RERANK_BATCH_SIZE = 128;
    constexpr size_t GPU_CHUNK_SIZE = 256;

    const std::string DEFAULT_HF_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2";

} // namespace Config

#endif // CONFIG_H