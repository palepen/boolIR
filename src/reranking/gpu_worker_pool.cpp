#include "reranking/gpu_worker_pool.h"
#include <iostream>

GpuWorkerPool::GpuWorkerPool(const std::string& model_path, const std::string& vocab_path, size_t num_workers)
    : model_path_(model_path), vocab_path_(vocab_path) {
    std::cout << "Initializing GPU Worker Pool with " << num_workers << " workers..." << std::endl;
    for (size_t i = 0; i < num_workers; ++i) {
        workers_.emplace_back(&GpuWorkerPool::worker_loop, this, i);
    }
}

GpuWorkerPool::~GpuWorkerPool() {
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        stop_ = true;
    }
    condition_.notify_all();
    for (std::thread &worker : workers_) {
        worker.join();
    }
}

std::future<std::vector<ScoredDocument>> GpuWorkerPool::submit_job(const std::string& query_text, const std::vector<Document>& candidates) {
    RerankJob job;
    job.query_text = query_text;
    job.candidates = candidates;
    auto future = job.promise.get_future();
    
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        job_queue_.push(std::move(job));
    }
    condition_.notify_one();
    return future;
}

void GpuWorkerPool::worker_loop(int worker_id) {
    // Each worker thread gets its own instance of the reranker.
    // The ONNX Runtime will manage GPU access between these instances.
    GpuNeuralReranker reranker(model_path_.c_str(), vocab_path_.c_str());
    
    while (true) {
        RerankJob job;
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            condition_.wait(lock, [this] { return stop_ || !job_queue_.empty(); });
            if (stop_ && job_queue_.empty()) {
                return;
            }
            job = std::move(job_queue_.front());
            job_queue_.pop();
        }

        try {
            auto results = reranker.rerank(job.query_text, job.candidates);
            job.promise.set_value(std::move(results));
        } catch (const std::exception& e) {
            std::cerr << "Worker " << worker_id << " caught exception: " << e.what() << std::endl;
            try {
                job.promise.set_exception(std::current_exception());
            } catch(...) {} // Ignore exceptions on setting exception
        }
    }
}