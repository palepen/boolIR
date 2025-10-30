#include "common/progress_bar.h"
#include <cmath>
#include <algorithm>


ProgressBar::ProgressBar(size_t total, const std::string& description, size_t width)
    : total_(total), current_(0), width_(width), description_(description), finished_(false) {
    start_time_ = std::chrono::high_resolution_clock::now();
    if (total_ > 0) {
        display();
    }
}

void ProgressBar::update(size_t increment) {
    current_ += increment;
    if (current_ > total_) {
        current_ = total_;
    }
    display();
}

void ProgressBar::set_progress(size_t current) {
    current_ = std::min(current, total_);
    display();
}

void ProgressBar::display() {
    if (finished_ || total_ == 0) {
        return;
    }
    
    // Calculate progress
    double fraction = static_cast<double>(current_) / total_;
    size_t filled = static_cast<size_t>(fraction * width_);
    
    // Calculate timing
    auto now = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(now - start_time_).count();
    double rate = (elapsed > 0) ? (current_ / elapsed) : 0.0;
    double eta = (rate > 0 && current_ < total_) ? ((total_ - current_) / rate) : 0.0;
    
    // Build progress bar
    std::ostringstream oss;
    oss << "\r";
    
    if (!description_.empty()) {
        oss << description_ << ": ";
    }
    
    // Progress bar
    oss << "[";
    for (size_t i = 0; i < width_; ++i) {
        if (i < filled) {
            oss << "=";
        } else if (i == filled && current_ < total_) {
            oss << ">";
        } else {
            oss << " ";
        }
    }
    oss << "] ";
    
    // Percentage
    oss << std::fixed << std::setprecision(1) << (fraction * 100.0) << "% ";
    
    // Count
    oss << "(" << current_ << "/" << total_ << ") ";
    
    // Rate and ETA
    if (rate > 0) {
        oss << format_rate(rate) << " ";
        if (current_ < total_) {
            oss << "ETA: " << format_time(eta);
        }
    }
    
    // Clear to end of line and flush
    oss << "    ";
    std::cout << oss.str() << std::flush;
    
    if (current_ >= total_) {
        std::cout << std::endl;
    }
}

void ProgressBar::finish() {
    if (finished_) {
        return;
    }
    
    current_ = total_;
    finished_ = true;
    
    auto now = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(now - start_time_).count();
    double rate = (elapsed > 0) ? (total_ / elapsed) : 0.0;
    
    std::ostringstream oss;
    oss << "\r";
    
    if (!description_.empty()) {
        oss << description_ << ": ";
    }
    
    // Completed bar
    oss << "[";
    for (size_t i = 0; i < width_; ++i) {
        oss << "=";
    }
    oss << "] ";
    
    oss << "100.0% ";
    oss << "(" << total_ << "/" << total_ << ") ";
    oss << format_rate(rate) << " ";
    oss << "Total: " << format_time(elapsed);
    oss << "    ";
    
    std::cout << oss.str() << std::endl;
}

std::string ProgressBar::format_time(double seconds) const {
    if (seconds < 60) {
        return std::to_string(static_cast<int>(seconds)) + "s";
    } else if (seconds < 3600) {
        int mins = static_cast<int>(seconds / 60);
        int secs = static_cast<int>(seconds) % 60;
        return std::to_string(mins) + "m " + std::to_string(secs) + "s";
    } else {
        int hours = static_cast<int>(seconds / 3600);
        int mins = static_cast<int>(seconds / 60) % 60;
        return std::to_string(hours) + "h " + std::to_string(mins) + "m";
    }
}

std::string ProgressBar::format_rate(double rate) const {
    if (rate < 1.0) {
        return std::to_string(static_cast<int>(rate * 60)) + "/min";
    } else if (rate < 1000) {
        return std::to_string(static_cast<int>(rate)) + "/s";
    } else {
        return std::to_string(static_cast<int>(rate / 1000)) + "k/s";
    }
}

// =====================================================================
// Spinner Implementation
// =====================================================================

const char* Spinner::frames_[] = {"|", "/", "-", "\\"};

Spinner::Spinner(const std::string& description)
    : description_(description), frame_(0) {
    start_time_ = std::chrono::high_resolution_clock::now();
    update();
}

void Spinner::update() {
    auto now = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(now - start_time_).count();
    
    std::ostringstream oss;
    oss << "\r";
    
    if (!description_.empty()) {
        oss << description_ << " ";
    }
    
    oss << frames_[frame_] << " ";
    oss << "(" << std::fixed << std::setprecision(1) << elapsed << "s)";
    oss << "    ";
    
    std::cout << oss.str() << std::flush;
    
    frame_ = (frame_ + 1) % 4;
}

void Spinner::finish(const std::string& final_message) {
    auto now = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(now - start_time_).count();
    
    std::ostringstream oss;
    oss << "\r";
    
    if (!description_.empty()) {
        oss << description_ << " ";
    }
    
    if (!final_message.empty()) {
        oss << final_message << " ";
    } else {
        oss << "Done! ";
    }
    
    oss << "(" << std::fixed << std::setprecision(2) << elapsed << "s)";
    oss << "    ";
    
    std::cout << oss.str() << std::endl;
}