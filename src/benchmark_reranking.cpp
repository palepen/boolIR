#include "reranking/neural_reranker.h"
#include "indexing/performance_monitor.h"
#include <iostream>
#include <iomanip>
#include <vector>

int main() {
    const char* model_path = "models/bert_model.onnx";
    std::cout << "--- Phase 3: Benchmarking Neural Re-ranking ---\n";
    std::cout << "This requires the ONNX model at '" << model_path << "' (run 'make model' first).\n\n";

    std::vector<Document> candidates;
    for (int i = 0; i < 100; ++i) {
        candidates.push_back({i, "document text content " + std::to_string(i)});
    }
    std::string query = "This is the user query";
    PerformanceMonitor perf;
    
    try {
        // Benchmark Sequential CPU Re-ranker
        std::cout << "Initializing CPU reranker...\n";
        SequentialNeuralReranker cpu_reranker(model_path);
        
        std::cout << "Running CPU reranking benchmark...\n";
        perf.start_timer("cpu_rerank_latency");
        auto cpu_results = cpu_reranker.rerank(query, candidates);
        perf.end_timer("cpu_rerank_latency");
        
        std::cout << "CPU reranking complete. Found " << cpu_results.size() << " results.\n\n";

        // Try GPU Re-ranker (optional)
        bool gpu_available = false;
        double gpu_time = 0.0;
        
        try {
            std::cout << "Attempting to initialize GPU reranker...\n";
            GpuNeuralReranker gpu_reranker(model_path, 32);
            gpu_available = true;
            
            std::cout << "Running GPU reranking benchmark...\n";
            perf.start_timer("gpu_rerank_latency");
            auto gpu_results = gpu_reranker.rerank(query, candidates);
            perf.end_timer("gpu_rerank_latency");
            
            gpu_time = perf.get_duration_ms("gpu_rerank_latency");
            std::cout << "GPU reranking complete. Found " << gpu_results.size() << " results.\n\n";
            
        } catch (const Ort::Exception& e) {
            std::cout << "\n[INFO] GPU reranking not available: " << e.what() << "\n";
            std::cout << "       This is normal if CUDA is not installed or no GPU is present.\n";
            std::cout << "       Continuing with CPU-only benchmarking...\n\n";
        }

        // --- Print Performance ---
        std::cout << "\n--- Re-ranking Benchmark Results (100 candidates) ---\n";
        std::cout << std::fixed << std::setprecision(3);
        double cpu_time = perf.get_duration_ms("cpu_rerank_latency");

        std::cout << std::left << std::setw(25) << "Strategy" << "Latency (ms)" << std::endl;
        std::cout << "------------------------------------------\n";
        std::cout << std::left << std::setw(25) << "Sequential CPU" << cpu_time << std::endl;
        
        if (gpu_available) {
            std::cout << std::left << std::setw(25) << "GPU (Batch of 32)" << gpu_time << std::endl;
            std::cout << "------------------------------------------\n";
            if (gpu_time > 0) {
                std::cout << "Speedup Factor: " << (cpu_time / gpu_time) << "x\n\n";
            }
        } else {
            std::cout << std::left << std::setw(25) << "GPU (Batch of 32)" << "N/A (CUDA not available)" << std::endl;
            std::cout << "------------------------------------------\n";
        }

    } catch (const Ort::Exception& e) {
        std::cerr << "\n[ERROR] ONNX Runtime Error: " << e.what() << std::endl;
        std::cerr << "Please ensure the model file exists at: " << model_path << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "\n[ERROR] " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}