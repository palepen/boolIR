#!/bin/bash
# OpenCilk Performance benchmarking script

echo "Running OpenCilk performance benchmarks..."
echo "============================================"

# Test different OpenCilk worker configurations
echo "Testing OpenCilk scalability with different worker counts..."

# Create benchmark results directory
mkdir -p results/benchmarks

# Test with different worker counts
for workers in 1 2 4 8 16 32; do
    echo ""
    echo "=== Testing with $workers OpenCilk workers ==="
    
    # Set OpenCilk worker count
    export CILK_NWORKERS=$workers
    
    echo "Building index with $workers workers..."
    /usr/bin/time -v ./boolean_retrieval --mode baseline --dataset docs --output results/benchmarks/perf_${workers}workers.txt 2> results/benchmarks/time_${workers}workers.txt
    
    # Extract timing information
    real_time=$(grep "Elapsed (wall clock)" results/benchmarks/time_${workers}workers.txt | awk '{print $8}')
    user_time=$(grep "User time" results/benchmarks/time_${workers}workers.txt | awk '{print $4}')
    sys_time=$(grep "System time" results/benchmarks/time_${workers}workers.txt | awk '{print $4}')
    max_memory=$(grep "Maximum resident set size" results/benchmarks/time_${workers}workers.txt | awk '{print $6}')
    
    echo "Results for $workers workers:"
    echo "  Wall time: $real_time"
    echo "  User time: $user_time"
    echo "  System time: $sys_time"
    echo "  Max memory: $max_memory KB"
    
    # Log to summary file
    echo "$workers,$real_time,$user_time,$sys_time,$max_memory" >> results/benchmarks/scalability_results.csv
done

echo ""
echo "Benchmark complete! Results saved to results/benchmarks/"
echo "Summary: results/benchmarks/scalability_results.csv"

# Generate scalability plot if Python is available
if command -v python3 &> /dev/null; then
    python3 scripts/plot_scalability.py results/benchmarks/scalability_results.csv
fi