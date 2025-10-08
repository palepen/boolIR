#ifndef NEURAL_RERANKER_H
#define NEURAL_RERANKER_H

#include "indexing/document.h"
#include <string>
#include <vector>
#include <memory>
#include <onnxruntime_cxx_api.h>

struct ScoredDocument
{
    unsigned int id;
    float score;
    bool operator<(const ScoredDocument &other) const { return score > other.score; }
};

class NeuralReranker
{
public:
    virtual ~NeuralReranker() = default;
    virtual std::vector<ScoredDocument> rerank(const std::string &query, const std::vector<Document> &candidates) = 0;
};

class SequentialNeuralReranker : public NeuralReranker
{
public:
    explicit SequentialNeuralReranker(const char *model_path);
    std::vector<ScoredDocument> rerank(const std::string &query, const std::vector<Document> &candidates) override;

private:
    Ort::Env env_;
    Ort::Session session_{nullptr};
    Ort::AllocatorWithDefaultOptions allocator_;
    std::vector<const char *> input_names_{"input_ids", "attention_mask"};
    std::vector<const char *> output_names_{"last_hidden_state"};

    std::vector<float> compute_embedding(const std::string &text);
};

class GpuNeuralReranker : public NeuralReranker
{
public:
    explicit GpuNeuralReranker(const char *model_path, size_t batch_size = 32);
    std::vector<ScoredDocument> rerank(const std::string &query, const std::vector<Document> &candidates) override;

private:
    Ort::Env env_;
    Ort::Session session_{nullptr};
    Ort::AllocatorWithDefaultOptions allocator_;
    size_t batch_size_;
    std::vector<const char *> input_names_{"input_ids", "attention_mask"};
    std::vector<const char *> output_names_{"last_hidden_state"};

    std::vector<float> compute_embedding(const std::string &text);
    std::vector<std::vector<float>> compute_batch_embeddings(const std::vector<std::string> &texts);
};


#endif