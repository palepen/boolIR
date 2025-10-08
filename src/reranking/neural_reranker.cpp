#include "reranking/neural_reranker.h"
#include <iostream>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <vector>

// Helper function for cosine similarity
float cosine_similarity(const std::vector<float> &v1, const std::vector<float> &v2)
{
    if (v1.empty() || v2.empty())
        return 0.0f;
    float dot_product = 0.0, norm1 = 0.0, norm2 = 0.0;
    for (size_t i = 0; i < v1.size(); ++i)
    {
        dot_product += v1[i] * v2[i];
        norm1 += v1[i] * v1[i];
        norm2 += v2[i] * v2[i];
    }
    if (norm1 == 0.0f || norm2 == 0.0f)
        return 0.0f;
    return dot_product / (std::sqrt(norm1) * std::sqrt(norm2));
}

// --- Helper function to simulate tokenization ---
// A real implementation would require a full WordPiece tokenizer library.
// This function creates a unique but deterministic placeholder tensor based on the text.
void simulate_tokenization(const std::string& text, int64_t seq_len, std::vector<int64_t>& out_input_ids, std::vector<int64_t>& out_attention_mask) {
    out_input_ids.assign(seq_len, 0); // Padding token
    out_attention_mask.assign(seq_len, 0);

    // Start with CLS token
    out_input_ids[0] = 101;
    out_attention_mask[0] = 1;
    
    // Simulate token IDs based on characters in the text
    size_t tokens_to_generate = std::min((size_t)seq_len - 2, text.length());
    for(size_t i = 0; i < tokens_to_generate; ++i) {
        out_input_ids[i + 1] = 1000 + (static_cast<int64_t>(text[i]) % 29000); // Simple hash to a vocab ID
        out_attention_mask[i + 1] = 1;
    }

    // End with SEP token
    out_input_ids[tokens_to_generate + 1] = 102;
    out_attention_mask[tokens_to_generate + 1] = 1;
}


// --- SequentialNeuralReranker Implementation ---

SequentialNeuralReranker::SequentialNeuralReranker(const char *model_path)
    : env_(ORT_LOGGING_LEVEL_WARNING, "cpu_reranker"),
      session_(env_, model_path, Ort::SessionOptions{})
{
    std::cout << "Loading CPU model from: " << model_path << std::endl;
    std::cout << "CPU model loaded successfully." << std::endl;
}

std::vector<ScoredDocument> SequentialNeuralReranker::rerank(
    const std::string &query, const std::vector<Document> &candidates)
{
    std::vector<float> query_embedding = compute_embedding(query);
    std::vector<ScoredDocument> ranked_results;

    for (const auto &doc : candidates)
    {
        std::vector<float> doc_embedding = compute_embedding(doc.content);
        float score = cosine_similarity(query_embedding, doc_embedding);
        ranked_results.push_back({doc.id, score});
    }

    std::sort(ranked_results.begin(), ranked_results.end());
    return ranked_results;
}

std::vector<float> SequentialNeuralReranker::compute_embedding(const std::string &text)
{
    const int64_t seq_len = 128;
    std::vector<int64_t> input_ids;
    std::vector<int64_t> attention_mask;
    simulate_tokenization(text, seq_len, input_ids, attention_mask);

    const int64_t input_shape[] = {1, seq_len};
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
        memory_info, input_ids.data(), input_ids.size(), input_shape, 2));
    input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
        memory_info, attention_mask.data(), attention_mask.size(), input_shape, 2));

    auto output_tensors = session_.Run(
        Ort::RunOptions{nullptr}, input_names_.data(), input_tensors.data(),
        input_tensors.size(), output_names_.data(), 1);

    float *floatarr = output_tensors[0].GetTensorMutableData<float>();
    auto shape_info = output_tensors[0].GetTensorTypeAndShapeInfo();
    
    auto shape = shape_info.GetShape();
    size_t hidden_size = shape[2]; 

    // Mean pooling
    std::vector<float> mean_pooled(hidden_size, 0.0f);
    for (size_t j = 0; j < seq_len; ++j) {
        if (attention_mask[j] == 1) { // Only pool valid tokens
            for (size_t k = 0; k < hidden_size; ++k) {
                mean_pooled[k] += floatarr[j * hidden_size + k];
            }
        }
    }
    size_t num_valid_tokens = std::accumulate(attention_mask.begin(), attention_mask.end(), 0);
    if (num_valid_tokens > 0) {
        for (size_t k = 0; k < hidden_size; ++k) {
            mean_pooled[k] /= num_valid_tokens;
        }
    }
    
    return mean_pooled;
}

// --- GpuNeuralReranker Implementation ---

GpuNeuralReranker::GpuNeuralReranker(const char *model_path, size_t batch_size)
    : env_(ORT_LOGGING_LEVEL_WARNING, "gpu_reranker"), 
      batch_size_(batch_size),
      session_(env_, model_path, []() {
          Ort::SessionOptions opts;
          OrtCUDAProviderOptions cuda_options{};
          opts.AppendExecutionProvider_CUDA(cuda_options);
          return opts;
      }())
{
    std::cout << "Loading GPU model from: " << model_path << std::endl;
    std::cout << "GPU model loaded successfully." << std::endl;
}

std::vector<ScoredDocument> GpuNeuralReranker::rerank(
    const std::string &query, const std::vector<Document> &candidates)
{
    std::vector<float> query_embedding = compute_embedding(query);
    std::vector<ScoredDocument> ranked_results;
    for (size_t i = 0; i < candidates.size(); i += batch_size_)
    {
        std::vector<std::string> batch_texts;
        std::vector<const Document *> batch_docs;
        size_t end = std::min(i + batch_size_, candidates.size());
        for (size_t j = i; j < end; ++j)
        {
            batch_texts.push_back(candidates[j].content);
            batch_docs.push_back(&candidates[j]);
        }
        std::vector<std::vector<float>> batch_embeddings = compute_batch_embeddings(batch_texts);
        for (size_t k = 0; k < batch_embeddings.size(); ++k)
        {
            float score = cosine_similarity(query_embedding, batch_embeddings[k]);
            ranked_results.push_back({batch_docs[k]->id, score});
        }
    }
    std::sort(ranked_results.begin(), ranked_results.end());
    return ranked_results;
}

std::vector<float> GpuNeuralReranker::compute_embedding(const std::string &text)
{
    return compute_batch_embeddings({text})[0];
}

std::vector<std::vector<float>> GpuNeuralReranker::compute_batch_embeddings(const std::vector<std::string> &texts)
{
    size_t current_batch_size = texts.size();
    const int64_t seq_len = 128;
    
    std::vector<int64_t> all_input_ids;
    std::vector<int64_t> all_attention_masks;
    all_input_ids.reserve(current_batch_size * seq_len);
    all_attention_masks.reserve(current_batch_size * seq_len);

    for(const auto& text : texts) {
        std::vector<int64_t> input_ids;
        std::vector<int64_t> attention_mask;
        simulate_tokenization(text, seq_len, input_ids, attention_mask);
        all_input_ids.insert(all_input_ids.end(), input_ids.begin(), input_ids.end());
        all_attention_masks.insert(all_attention_masks.end(), attention_mask.begin(), attention_mask.end());
    }

    int64_t input_shape[] = {(int64_t)current_batch_size, seq_len};
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
        memory_info, all_input_ids.data(), all_input_ids.size(), input_shape, 2));
    input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
        memory_info, all_attention_masks.data(), all_attention_masks.size(), input_shape, 2));

    auto output_tensors = session_.Run(
        Ort::RunOptions{nullptr}, input_names_.data(), input_tensors.data(), input_tensors.size(), output_names_.data(), 1);
    
    float *floatarr = output_tensors[0].GetTensorMutableData<float>();
    auto shape_info = output_tensors[0].GetTensorTypeAndShapeInfo();
    
    auto shape = shape_info.GetShape();
    size_t hidden_size = shape[2];
    
    std::vector<std::vector<float>> results;
    results.reserve(current_batch_size);

    for (size_t i = 0; i < current_batch_size; ++i)
    {
        std::vector<float> mean_pooled(hidden_size, 0.0f);
        size_t num_valid_tokens = 0;
        for (size_t j = 0; j < seq_len; ++j)
        {
            size_t batch_offset = i * seq_len;
            if (all_attention_masks[batch_offset + j] == 1) {
                num_valid_tokens++;
                 for (size_t k = 0; k < hidden_size; ++k)
                {
                    mean_pooled[k] += floatarr[(batch_offset + j) * hidden_size + k];
                }
            }
        }
        if (num_valid_tokens > 0) {
            for (size_t k = 0; k < hidden_size; ++k) {
                mean_pooled[k] /= num_valid_tokens;
            }
        }
        results.push_back(mean_pooled);
    }
    return results;
}