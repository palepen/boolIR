#ifndef PROGRESS_BAR_H
#define PROGRESS_BAR_H

#include <string>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <sstream>

/**
 * Simple progress bar utility for terminal output
 * Thread-safe for single-threaded sequential operations
 */
class ProgressBar {
public:

    ProgressBar(size_t total, const std::string& description = "", size_t width = 50);
    
    
    void update(size_t increment = 1);
    

    void set_progress(size_t current);
    

    void finish();
    

    bool is_finished() const { return current_ >= total_; }

private:
    size_t total_;
    size_t current_;
    size_t width_;
    std::string description_;
    std::chrono::high_resolution_clock::time_point start_time_;
    bool finished_;
    
    void display();
    std::string format_time(double seconds) const;
    std::string format_rate(double rate) const;
};

/**
 * Spinner for indefinite progress (when total is unknown)
 */
class Spinner {
public:
    explicit Spinner(const std::string& description = "");
    

    void update();
    

    void finish(const std::string& final_message = "");

private:
    std::string description_;
    size_t frame_;
    std::chrono::high_resolution_clock::time_point start_time_;
    static const char* frames_[];
};

#endif // PROGRESS_BAR_H