# --- Compiler and Flags ---
CXX = /opt/opencilk/bin/clang++
CUDA_PATH ?= /usr/local/cuda

CXXFLAGS = -std=c++17 -fopencilk -O3 -pthread -Iinclude -I/opt/onnxruntime/include -I$(CUDA_PATH)/include
LDFLAGS = -L/opt/opencilk/lib -L/opt/onnxruntime/lib -lonnxruntime -no-pie \
          -Wl,-rpath,/opt/opencilk/lib,-rpath,/opt/onnxruntime/lib -L$(CUDA_PATH)/lib64 -lcudart

# --- Directories ---
SRC_DIR = src
OBJ_DIR = build/obj
BIN_DIR = build/bin
RESULTS_DIR = results
INDEX_DIR = index
MODEL_DIR = models
DATA_DIR = data

# --- Benchmark Configuration ---
CPU_WORKER_COUNTS = 1 2 4 8
LOG_FILE = $(RESULTS_DIR)/full_benchmark_output.log
CSV_FILE = $(RESULTS_DIR)/all_benchmarks.csv

# --- Target Executable ---
TARGET = $(BIN_DIR)/full_system_benchmark

# --- Source Files ---
CORE_SRCS = $(SRC_DIR)/main.cpp \
            $(SRC_DIR)/system_controller.cpp \
            $(SRC_DIR)/benchmark_suite.cpp \
            $(SRC_DIR)/data_loader.cpp \
            $(SRC_DIR)/document_store.cpp

INDEX_SRCS = $(SRC_DIR)/indexing/bsbi_indexer.cpp \
             $(SRC_DIR)/indexing/posting_list.cpp \
             $(SRC_DIR)/indexing/performance_monitor.cpp

# MODIFIED: Updated the retrieval sources
RETRIEVAL_SRCS = $(SRC_DIR)/retrieval/retrieval_set.cpp \
                 $(SRC_DIR)/retrieval/dynamic_retriever.cpp \
                 $(SRC_DIR)/retrieval/query_expander.cpp \
                 $(SRC_DIR)/retrieval/query_preprocessor.cpp

RERANK_SRCS = $(SRC_DIR)/reranking/neural_reranker.cpp
TOKEN_SRCS = $(SRC_DIR)/tokenizer/wordpiece_tokenizer.cpp
EVAL_SRCS = $(SRC_DIR)/evaluation/evaluator.cpp

ALL_SRCS = $(CORE_SRCS) $(INDEX_SRCS) $(RETRIEVAL_SRCS) $(RERANK_SRCS) $(TOKEN_SRCS) $(EVAL_SRCS)
ALL_OBJS = $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(ALL_SRCS))

.PHONY: all clean model dirs index run benchmark plot

all: dirs $(TARGET)

dirs:
	@mkdir -p $(OBJ_DIR)/indexing $(OBJ_DIR)/retrieval $(OBJ_DIR)/reranking $(OBJ_DIR)/tokenizer $(OBJ_DIR)/evaluation
	@mkdir -p $(BIN_DIR) $(RESULTS_DIR) $(INDEX_DIR)

model:
	@echo "Exporting BERT cross-encoder model to ONNX format..."
	@python3 scripts/export_model.py

index: $(TARGET)
	@echo "Building sharded on-disk index (64 shards)..."
	@./$(TARGET) --build-index --shards 64

$(TARGET): $(ALL_OBJS)
	@echo "[LINK] Building executable: $@"
	@$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@
	@echo "Build complete: $@"

# --- Main Benchmark & Run Targets (Simplified) ---
benchmark: $(TARGET)
	@echo "Running integrated query performance benchmarks..."
	@echo "Results will be logged to $(LOG_FILE) and $(CSV_FILE)"
	@rm -f $(LOG_FILE) $(CSV_FILE)
	@for workers in $(CPU_WORKER_COUNTS); do \
		echo "\n[BENCHMARK] Running with $$workers CPU workers..." | tee -a $(LOG_FILE); \
		CILK_NWORKERS=$$workers ./$(TARGET) --benchmark --label "Sharded_$$workers-cpu" --cpu-workers $$workers | tee -a $(LOG_FILE); \
	done
	@echo "\nAll benchmarks completed. Consolidated results in $(CSV_FILE)"

plot:
	@echo "Generating performance plots from benchmark results..."
	@python3 scripts/evaluation_metrics.py --results $(CSV_FILE)

run: $(TARGET)
	@echo "Running interactive mode..."
	@./$(TARGET) --interactive

# --- Object File Compilation Rules ---
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@echo "[CXX] Compiling $<"
	@$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	@echo "Cleaning build artifacts"
	@rm -rf build
	@echo "Clean complete."