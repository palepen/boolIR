#ifndef PERFORMANCE_MONITOR_H
#define PERFORMANCE_MONITOR_H
#include <string>
#include <chrono>
#include <unordered_map>
#include <vector>
#include <mutex>
#include <map>

struct IndexingMetrics {
    double indexing_time_ms  = 0.0;
    size_t memory_peak_mb = 0;
    double throughput_docs_per_sec = 0.0;
    std::map<int, double> core_scaling_factor;
};

class PerformanceMonitor {
    public:
        void start_timer(const std::string &label);
        void end_timer(const std::string &label);
        void print_summary() const;

        double get_duration_ms(const std::string &label) const;

    private:
        struct TimingData
        {
            std::chrono::high_resolution_clock::time_point start_time;
            double total_duration_ms = 0.0;
        };

        std::unordered_map<std::string, TimingData> timings_;
        mutable std::mutex mtx_;        
        
    };
#endif