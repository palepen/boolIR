#include "indexing/performance_monitor.h"
#include <iostream>
#include <iomanip>

void PerformanceMonitor::start_timer(const std::string& label) {
    std::lock_guard<std::mutex> lock(mtx_);
    timings_[label].start_time = std::chrono::high_resolution_clock::now();
}

void PerformanceMonitor::end_timer(const std::string& label) {
    std::lock_guard<std::mutex> lock(mtx_);
    auto it = timings_.find(label);
    if (it != timings_.end()) {
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end_time - it->second.start_time;
        it->second.total_duration_ms += duration.count();
    }
}

double PerformanceMonitor::get_duration_ms(const std::string& label) const {
    std::lock_guard<std::mutex> lock(mtx_);
    auto it = timings_.find(label);
    if (it != timings_.end()) {
        return it->second.total_duration_ms;
    }
    return 0.0;
}

void PerformanceMonitor::print_summary() const {
    std::lock_guard<std::mutex> lock(mtx_);
    std::cout << "\n--- Performance Summary ---\n";
    for (const auto& pair : timings_) {
        std::cout << std::left << std::setw(30) << pair.first << ": "
                  << std::fixed << std::setprecision(3) << pair.second.total_duration_ms << " ms\n";
    }
    std::cout << "---------------------------\n";
}