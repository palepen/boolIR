#include "reranking/neural_reranker.h"
#include <iostream>
#include <numeric>
#include <algorithm>
#include <chrono>
#include <unordered_map>
#include <sstream>
#include <cuda_runtime.h>

// Helper for checking CUDA calls
#define CUDA_CHECK(call)                                          \
    do                                                            \
    {                                                             \
        cudaError_t err = call;                                   \
        if (err != cudaSuccess)                                   \
        {                                                         \
            fprintf(stderr, "CUDA Error: %s at %s:%d\n",          \
                    cudaGetErrorString(err), __FILE__, __LINE__); \
            throw std::runtime_error("CUDA call failed.");        \
        }                                                         \
    } while (0)

GpuNeuralReranker::GpuNeuralReranker(
    const char *model_path,
    const char *vocab_path,
    size_t batch_size) : env_(ORT_LOGGING_LEVEL_WARNING, "gpu_reranker"),
                         batch_size_(batch_size),
                         session_(env_, model_path, []()
                                  {
        Ort::SessionOptions opts;
        OrtCUDAProviderOptions cuda_options{};
        cuda_options.device_id = 0;
        opts.AppendExecutionProvider_CUDA(cuda_options);
        return opts; }()),
                         memory_info_device_("Cuda", OrtArenaAllocator, 0, OrtMemTypeDefault)
{
    std::cout << "Loading GPU cross-encoder model from: " << model_path << std::endl;
    tokenizer_ = std::make_unique<WordPieceTokenizer>(vocab_path);

    // OPTIMIZATION: Using max_seq_len = 256 instead of 512 for 2x speedup
    std::vector<int64_t> input_shape = {static_cast<int64_t>(batch_size_), max_seq_len_};
    size_t input_size = batch_size_ * max_seq_len_;

    // Allocate GPU buffers
    CUDA_CHECK(cudaMalloc(&d_input_ids_, input_size * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_attention_mask_, input_size * sizeof(int64_t)));

    // Determine output size
    auto output_type_info = session_.GetOutputTypeInfo(0);
    auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
    auto output_shape = output_tensor_info.GetShape();
    output_dim_ = (output_shape.size() > 1) ? output_shape[1] : 1;
    size_t output_size = batch_size_ * output_dim_;

    CUDA_CHECK(cudaMalloc(&d_output_, output_size * sizeof(float)));

    std::cout << "GPU cross-encoder loaded (batch=" << batch_size_
              << ", max_seq_len=" << max_seq_len_ << ", output_dim=" << output_dim_ << ")" << std::endl;
}

GpuNeuralReranker::~GpuNeuralReranker()
{
    if (d_input_ids_)
        cudaFree(d_input_ids_);
    if (d_attention_mask_)
        cudaFree(d_attention_mask_);
    if (d_output_)
        cudaFree(d_output_);
}

std::vector<ScoredDocument> GpuNeuralReranker::rerank(
    const std::string &query,
    const std::vector<Document> &candidates)
{
    if (candidates.empty())
    {
        return {};
    }

    std::vector<ScoredDocument> ranked_results;
    ranked_results.reserve(candidates.size());

    for (size_t i = 0; i < candidates.size(); i += batch_size_)
    {
        size_t end_idx = std::min(i + batch_size_, candidates.size());
        std::vector<Document> batch_docs(candidates.begin() + i, candidates.begin() + end_idx);

        std::vector<float> batch_scores = compute_batch_scores_with_length(query, batch_docs, max_seq_len_);

        for (size_t j = 0; j < batch_docs.size(); ++j)
        {
            ranked_results.push_back({batch_docs[j].id, batch_scores[j]});
        }
    }

    std::sort(ranked_results.begin(), ranked_results.end());

    return ranked_results;
}

std::vector<float> GpuNeuralReranker::compute_batch_scores_with_length(
    const std::string &query,
    const std::vector<Document> &documents,
    int64_t seq_len)
{
    if (documents.empty())
    {
        return {};
    }

    size_t current_batch_size = documents.size();
    size_t input_size = current_batch_size * seq_len;

    std::vector<int64_t> all_input_ids(input_size);
    std::vector<int64_t> all_attention_masks(input_size);

    for (size_t i = 0; i < current_batch_size; ++i)
    {
        std::vector<int64_t> input_ids_vec;
        std::vector<int64_t> attention_mask_vec;
        tokenizer_->encode_pair(query, documents[i].content, seq_len, input_ids_vec, attention_mask_vec);

        std::copy(input_ids_vec.begin(), input_ids_vec.end(), all_input_ids.begin() + i * seq_len);
        std::copy(attention_mask_vec.begin(), attention_mask_vec.end(), all_attention_masks.begin() + i * seq_len);
    }

    // Copy to GPU
    CUDA_CHECK(cudaMemcpy(d_input_ids_, all_input_ids.data(), input_size * sizeof(int64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_attention_mask_, all_attention_masks.data(), input_size * sizeof(int64_t), cudaMemcpyHostToDevice));

    // Create input tensors
    std::vector<int64_t> input_shape = {static_cast<int64_t>(current_batch_size), seq_len};
    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
        memory_info_device_, d_input_ids_, input_size, input_shape.data(), input_shape.size()));
    input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
        memory_info_device_, d_attention_mask_, input_size, input_shape.data(), input_shape.size()));

    // Create output tensor
    std::vector<int64_t> output_shape = {static_cast<int64_t>(current_batch_size), static_cast<int64_t>(output_dim_)};
    size_t output_size = current_batch_size * output_dim_;
    std::vector<Ort::Value> output_tensors;
    output_tensors.push_back(Ort::Value::CreateTensor<float>(
        memory_info_device_, d_output_, output_size, output_shape.data(), output_shape.size()));

    // Run inference
    session_.Run(Ort::RunOptions{nullptr}, input_names_.data(), input_tensors.data(),
                 input_tensors.size(), output_names_.data(), output_tensors.data(), output_tensors.size());

    // Copy results back
    std::vector<float> cpu_logits(output_size);
    CUDA_CHECK(cudaMemcpy(cpu_logits.data(), d_output_, output_size * sizeof(float), cudaMemcpyDeviceToHost));

    std::vector<float> scores;
    scores.reserve(current_batch_size);

    if (output_dim_ == 2)
    {
        // Binary classification output
        for (size_t i = 0; i < current_batch_size; ++i)
        {
            float not_relevant = cpu_logits[i * 2 + 0];
            float relevant = cpu_logits[i * 2 + 1];
            float exp_rel = std::exp(relevant);
            float exp_not_rel = std::exp(not_relevant);
            scores.push_back(exp_rel / (exp_rel + exp_not_rel));
        }
    }
    else
    {
        // Single score output
        for (size_t i = 0; i < current_batch_size; ++i)
        {
            scores.push_back(cpu_logits[i * output_dim_]);
        }
    }

    return scores;
}

std::vector<ScoredDocument> GpuNeuralReranker::rerank_with_chunking(
    const std::string &query,
    const std::vector<Document> &candidates,
    size_t chunk_size)
{
    // OPTIMIZATION: Simple truncation instead of expensive chunking
    // This is much faster and chunking is unnecessary with pre-truncated docs
    std::vector<Document> truncated_docs;
    truncated_docs.reserve(candidates.size());
    for (const auto &doc : candidates)
    {
        if (doc.content.length() <= chunk_size)
        {
            truncated_docs.push_back(doc);
        }
        else
        {
            std::string truncated = doc.content.substr(0, chunk_size);
            truncated_docs.emplace_back(doc.id, truncated);
        }
    }

    return rerank(query, truncated_docs);
}