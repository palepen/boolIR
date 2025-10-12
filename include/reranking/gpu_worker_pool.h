#ifndef GPU_WORKER_POOL_H
#define GPU_WORKER_POOL_H

#include "reranking/neural_reranker.h"
#include "indexing/document.h"
#include <vector>
#include <string>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <future>
#include <functional>

// Represents a single reranking job to be placed on the queue.

/**
 * @class GpuWorkerPool
 * @brief Manages a fixed-size pool of threads to process reranking jobs on the GPU.
 * This implements a producer-consumer pattern to control the flow of data to the GPU.
 */
class GpuWorkerPool {
public:
    /**
     * @param model_path Path to the ONNX model.
     * @param vocab_path Path to the vocabulary file.
     * @param num_workers The number of dedicated GPU worker threads to create (e.g., 2 or 4).
     */
    GpuWorkerPool(const std::string& model_path, const std::string& vocab_path, size_t num_workers = 2);
    ~GpuWorkerPool();

    /**
     * @brief Submits a new reranking job to the queue for a worker to pick up.
     * @return A std::future that will eventually contain the reranked results.
     */
    std::future<std::vector<ScoredDocument>> submit_job(const std::string& query_text, const std::vector<Document>& candidates);

private:
    void worker_loop(int worker_id);

    std::vector<std::thread> workers_;
    std::queue<RerankJob> job_queue_;
    std::mutex queue_mutex_;
    std::condition_variable condition_;
    bool stop_ = false;

    std::string model_path_;
    std::string vocab_path_;
};

#endif // GPU_WORKER_POOL_H