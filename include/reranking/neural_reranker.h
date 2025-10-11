#ifndef NEURAL_RERANKER_H
#define NEURAL_RERANKER_H

#include "indexing/document.h"
#include "tokenizer/wordpiece_tokenizer.h"
#include <onnxruntime_cxx_api.h>
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

    std::vector<ScoredDocument> rerank(
        const std::string& query,
        const std::vector<Document>& candidates
    );

private:
    std::vector<float> compute_batch_scores(
        const std::string& query,
        const std::vector<Document>& documents
    );

    Ort::Env env_;
    Ort::Session session_;
    std::unique_ptr<WordPieceTokenizer> tokenizer_;
    size_t batch_size_;
    const int64_t max_seq_len_ = 512;

    std::vector<const char*> input_names_{"input_ids", "attention_mask"};
    std::vector<const char*> output_names_{"logits"};
};

#endif // NEURAL_RERANKER_H