#ifndef NEURAL_RERANKER_H
#define NEURAL_RERANKER_H

#include "indexing/document.h"
#include "tokenizer/wordpiece_tokenizer.h"
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>
#include <memory>

// ScoredDocument for ranking results
struct ScoredDocument {
    unsigned int id;
    float score;
    bool operator<(const ScoredDocument &other) const {
        return score > other.score;
    }
};

// Helper struct for the chunking method
struct DocumentChunk {
    unsigned int doc_id;
    std::string content;
};

class GpuNeuralReranker {
public:
    // Updated constructor with larger default batch size
    GpuNeuralReranker(const char* model_path, const char* vocab_path, size_t batch_size = 32);
    ~GpuNeuralReranker();

    std::vector<ScoredDocument> rerank(
        const std::string& query,
        const std::vector<Document>& candidates
    );

    std::vector<ScoredDocument> rerank_with_chunking(
        const std::string& query,
        const std::vector<Document>& candidates,
        size_t chunk_size = 300
    );

private:
    std::vector<float> compute_batch_scores_with_length(
        const std::string& query,
        const std::vector<Document>& documents,
        int64_t seq_len
    );

    Ort::Env env_;
    Ort::Session session_;
    std::unique_ptr<WordPieceTokenizer> tokenizer_;
    size_t batch_size_;
    // OPTIMIZATION: Reduced from 512 to 256 for 2x speedup
    const int64_t max_seq_len_ = 256;

    std::vector<const char*> input_names_{"input_ids", "attention_mask"};
    std::vector<const char*> output_names_{"logits"};

    // GPU memory pointers
    Ort::MemoryInfo memory_info_device_;
    int64_t* d_input_ids_ = nullptr;
    int64_t* d_attention_mask_ = nullptr;
    float* d_output_ = nullptr;
    int64_t output_dim_ = 1;
};

#endif // NEURAL_RERANKER_H