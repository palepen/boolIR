#ifndef NEURAL_RERANKER_H
#define NEURAL_RERANKER_H

#include "indexing/document.h"
#include "tokenizer/wordpiece_tokenizer.h"
#include <torch/script.h>
#include <vector>
#include <string>
#include <memory>

struct ScoredDocument {
    unsigned int id;
    float score;
    bool operator<(const ScoredDocument &other) const {
        return score > other.score;
    }
};

class GpuNeuralReranker {
public:
    GpuNeuralReranker(const char* model_path, const char* vocab_path, size_t batch_size = 32);
    
    std::vector<ScoredDocument> rerank_with_chunking(
        const std::string& query,
        const std::vector<Document>& candidates,
        size_t chunk_size = 200
    );

private:
    std::vector<ScoredDocument> rerank_batch(
        const std::string& query,
        const std::vector<Document>& batch_docs
    );

    torch::jit::script::Module module_;
    torch::Device device_;
    std::unique_ptr<WordPieceTokenizer> tokenizer_;
    size_t batch_size_;
    const int64_t max_seq_len_ = 512;

    // --- NEW: Pre-allocated GPU tensors ---
    torch::Tensor input_ids_gpu_;
    torch::Tensor attention_mask_gpu_;
};

#endif // NEURAL_RERANKER_H