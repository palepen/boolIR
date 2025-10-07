# --- Compiler and Flags ---
CXX = /opt/opencilk/bin/clang++
# MODIFIED: Point to the standard system include path for ONNX Runtime
CXXFLAGS = -std=c++17 -fopencilk -O3 -pthread -Iinclude -I/opt/onnxruntime/include
# MODIFIED: Point to the standard system library path for ONNX Runtime
LDFLAGS = -L/opt/opencilk/lib -L/opt/onnxruntime/lib -lonnxruntime \
          -Wl,-rpath,/opt/opencilk/lib,-rpath,/opt/onnxruntime/lib

# --- Python Environment ---
# MODIFIED: Use the system's python3 directly
PYTHON = python3

# --- Executable Targets ---
RERANKING_TARGET = reranking_benchmark

# --- Source Files ---
# All necessary source files for the reranking benchmark
RERANKING_SRCS = src/benchmark_reranking.cpp \
                 src/reranking/neural_reranker.cpp \
                 src/indexing/performance_monitor.cpp

# --- Build Rules ---
# MODIFIED: Removed 'setup' from .PHONY
.PHONY: all clean reranking model

# Default action: build the Phase 3 executable
all: $(RERANKING_TARGET)

reranking: $(RERANKING_TARGET)

# REMOVED: The entire 'setup' rule for the virtual environment is no longer needed.

# Rule to export the BERT model to ONNX format
# MODIFIED: Removed the 'setup' dependency
model:
	@echo "Exporting BERT model to ONNX format..."
	$(PYTHON) scripts/export_model.py

# Rule to build the Phase 3 reranking benchmark
$(RERANKING_TARGET): $(RERANKING_SRCS)
	@echo "Building application: $@"
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@
	@echo "Build complete."

# Rule to clean all generated files
clean:
	@echo "Cleaning up..."
	rm -f $(RERANKING_TARGET)
	rm -rf models
