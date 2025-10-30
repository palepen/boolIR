#include "reranking/neural_reranker.h"
#include <torch/torch.h>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <stdexcept>

GpuNeuralReranker::GpuNeuralReranker(
    const char *model_path,
    const char *vocab_path,
    size_t batch_size)
    : device_(torch::kCUDA),
      batch_size_(batch_size)
{
    std::cout << "Loading TorchScript cross-encoder model from: " << model_path << std::endl;
    tokenizer_ = std::make_unique<WordPieceTokenizer>(vocab_path);

    try
    {
        module_ = torch::jit::load(model_path);
        module_.to(device_);
        module_.eval();
    }
    catch (const c10::Error &e)
    {
        throw std::runtime_error("Error loading TorchScript model: " + std::string(e.what()));
    }

    auto tensor_options = torch::TensorOptions().dtype(torch::kLong).device(device_);
    input_ids_gpu_ = torch::zeros({(long)batch_size_, max_seq_len_}, tensor_options);
    attention_mask_gpu_ = torch::zeros({(long)batch_size_, max_seq_len_}, tensor_options);

    std::cout << "GPU cross-encoder loaded via LibTorch (batch=" << batch_size_
              << ", max_seq_len=" << max_seq_len_ << ")" << std::endl;
}

std::vector<ScoredDocument> GpuNeuralReranker::rerank_batch(
    const std::string &query,
    const std::vector<Document> &batch_docs)
{
    if (batch_docs.empty())
        return {};

    torch::NoGradGuard no_grad;

    size_t current_batch_size = batch_docs.size();

    std::vector<int64_t> all_input_ids;
    std::vector<int64_t> all_attention_masks;
    all_input_ids.reserve(current_batch_size * max_seq_len_);
    all_attention_masks.reserve(current_batch_size * max_seq_len_);

    for (const auto &doc : batch_docs)
    {
        std::vector<int64_t> input_ids_vec;
        std::vector<int64_t> attention_mask_vec;
        tokenizer_->encode_pair(query, doc.content, max_seq_len_, input_ids_vec, attention_mask_vec);
        all_input_ids.insert(all_input_ids.end(), input_ids_vec.begin(), input_ids_vec.end());
        all_attention_masks.insert(all_attention_masks.end(), attention_mask_vec.begin(), attention_mask_vec.end());
    }

    // 1. Create temporary CPU tensors that POINT to the vector data (no copy)
    auto cpu_options = torch::TensorOptions().dtype(torch::kLong);
    torch::Tensor input_ids_cpu = torch::from_blob(all_input_ids.data(), {(long)current_batch_size, max_seq_len_}, cpu_options);
    torch::Tensor attention_mask_cpu = torch::from_blob(all_attention_masks.data(), {(long)current_batch_size, max_seq_len_}, cpu_options);

    // 2. Use a slice of the pre-allocated GPU tensor for the current batch size
    auto input_ids_view = input_ids_gpu_.slice(0, 0, current_batch_size);
    auto attention_mask_view = attention_mask_gpu_.slice(0, 0, current_batch_size);

    // 3. Perform an efficient copy from the CPU tensor to the GPU tensor's view
    input_ids_view.copy_(input_ids_cpu);
    attention_mask_view.copy_(attention_mask_cpu);

    // Prepare inputs for the model using the views
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input_ids_view);
    inputs.push_back(attention_mask_view);

    at::Tensor output_tensor = module_.forward(inputs).toTensor();

    // ... rest of the function remains the same ...
    output_tensor = output_tensor.to(torch::kCPU);
    auto output_accessor = output_tensor.accessor<float, 2>();

    std::vector<ScoredDocument> results;
    results.reserve(current_batch_size);
    for (size_t i = 0; i < current_batch_size; ++i)
    {
        results.push_back({batch_docs[i].id, output_accessor[i][0]});
    }
    return results;
}

std::vector<ScoredDocument> GpuNeuralReranker::rerank_with_chunking(
    const std::string &query,
    const std::vector<Document> &candidates,
    size_t chunk_size)
{
    if (candidates.empty())
        return {};

    std::vector<ScoredDocument> all_ranked_results;
    all_ranked_results.reserve(candidates.size());

    for (size_t i = 0; i < candidates.size(); i += batch_size_)
    {
        size_t end_idx = std::min(i + batch_size_, candidates.size());
        std::vector<Document> batch_docs(candidates.begin() + i, candidates.begin() + end_idx);
        auto batch_results = rerank_batch(query, batch_docs);
        all_ranked_results.insert(all_ranked_results.end(), batch_results.begin(), batch_results.end());
    }

    std::sort(all_ranked_results.begin(), all_ranked_results.end());
    return all_ranked_results;
}