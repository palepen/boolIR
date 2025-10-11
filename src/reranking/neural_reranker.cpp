#include "reranking/neural_reranker.h"
#include <iostream>
#include <numeric>
#include <algorithm>
#include <chrono>

GpuNeuralReranker::GpuNeuralReranker(
    const char* model_path,
    const char* vocab_path,
    size_t batch_size
) : env_(ORT_LOGGING_LEVEL_WARNING, "gpu_reranker"),
    batch_size_(batch_size),
    session_(env_, model_path, []() {
        Ort::SessionOptions opts;
        OrtCUDAProviderOptions cuda_options{};
        opts.AppendExecutionProvider_CUDA(cuda_options);
        return opts;
    }()) {
    
    std::cout << "Loading GPU cross-encoder model from: " << model_path << std::endl;
    tokenizer_ = std::make_unique<WordPieceTokenizer>(vocab_path);
    std::cout << "GPU cross-encoder and tokenizer loaded successfully." << std::endl;
}

std::vector<ScoredDocument> GpuNeuralReranker::rerank(
    const std::string& query,
    const std::vector<Document>& candidates
) {
    if (candidates.empty()) {
        return {};
    }
    
    std::vector<ScoredDocument> ranked_results;
    ranked_results.reserve(candidates.size());

    // Process candidates in batches
    for (size_t i = 0; i < candidates.size(); i += batch_size_) {
        size_t end_idx = std::min(i + batch_size_, candidates.size());
        std::vector<Document> batch_docs(candidates.begin() + i, candidates.begin() + end_idx);

        // Get scores for the current batch
        std::vector<float> batch_scores = compute_batch_scores(query, batch_docs);

        // Add to results
        for (size_t j = 0; j < batch_docs.size(); ++j) {
            ranked_results.push_back({batch_docs[j].id, batch_scores[j]});
        }
    }
    
    // Sort all results by score (descending)
    std::sort(ranked_results.begin(), ranked_results.end());
    
    return ranked_results;
}

std::vector<float> GpuNeuralReranker::compute_batch_scores(
    const std::string& query,
    const std::vector<Document>& documents
) {
    if (documents.empty()) {
        return {};
    }

    size_t current_batch_size = documents.size();
    
    // Tokenize all query-document pairs
    std::vector<int64_t> all_input_ids;
    std::vector<int64_t> all_attention_masks;
    all_input_ids.reserve(current_batch_size * max_seq_len_);
    all_attention_masks.reserve(current_batch_size * max_seq_len_);

    for (const auto& doc : documents) {
        std::vector<int64_t> input_ids;
        std::vector<int64_t> attention_mask;
        tokenizer_->encode_pair(query, doc.content, max_seq_len_, input_ids, attention_mask);
        
        all_input_ids.insert(all_input_ids.end(), input_ids.begin(), input_ids.end());
        all_attention_masks.insert(all_attention_masks.end(), attention_mask.begin(), attention_mask.end());
    }

    // Create batched input tensors for ONNX Runtime
    int64_t input_shape[] = {static_cast<int64_t>(current_batch_size), max_seq_len_};
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
        memory_info, all_input_ids.data(), all_input_ids.size(), input_shape, 2));
    input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
        memory_info, all_attention_masks.data(), all_attention_masks.size(), input_shape, 2));

    // Run batch inference on GPU
    auto output_tensors = session_.Run(
        Ort::RunOptions{nullptr},
        input_names_.data(),
        input_tensors.data(),
        input_tensors.size(),
        output_names_.data(),
        1
    );

    // Extract scores from the 'logits' output
    float* logits = output_tensors[0].GetTensorMutableData<float>();
    std::vector<float> scores(logits, logits + current_batch_size);

    return scores;
}