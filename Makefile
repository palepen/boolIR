# --- Compiler and Flags ---
CXX = /opt/opencilk/bin/clang++
# Allow CUDA_PATH to be overridden. Defaults to a common location.
CUDA_PATH ?= /usr/local/cuda

# Add the CUDA include path to CXXFLAGS
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
            $(SRC_DIR)/data_loader.cpp

INDEX_SRCS = $(SRC_DIR)/indexing/bsbi_indexer.cpp \
             $(SRC_DIR)/indexing/posting_list.cpp \
             $(SRC_DIR)/indexing/performance_monitor.cpp

RETRIEVAL_SRCS = $(SRC_DIR)/retrieval/retrieval_set.cpp \
                 $(SRC_DIR)/retrieval/optimized_parallel_retrieval.cpp \
                 $(SRC_DIR)/retrieval/query_expander.cpp \
                 $(SRC_DIR)/retrieval/query_preprocessor.cpp \
                 $(SRC_DIR)/retrieval/pre_ranker.cpp

RERANK_SRCS = $(SRC_DIR)/reranking/neural_reranker.cpp

TOKEN_SRCS = $(SRC_DIR)/tokenizer/wordpiece_tokenizer.cpp

EVAL_SRCS = $(SRC_DIR)/evaluation/evaluator.cpp

ALL_SRCS = $(CORE_SRCS) $(INDEX_SRCS) $(RETRIEVAL_SRCS) $(RERANK_SRCS) $(TOKEN_SRCS) $(EVAL_SRCS)
ALL_OBJS = $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(ALL_SRCS))

.PHONY: all clean model dirs index run benchmark benchmark-all benchmark-indexing plot

all: dirs $(TARGET)

dirs:
	@mkdir -p $(OBJ_DIR)/indexing $(OBJ_DIR)/retrieval $(OBJ_DIR)/reranking $(OBJ_DIR)/tokenizer $(OBJ_DIR)/evaluation
	@mkdir -p $(BIN_DIR) $(RESULTS_DIR) $(INDEX_DIR)

model:
	@echo "Exporting BERT cross-encoder model to ONNX format..."
	python3 scripts/export_model.py

index: $(TARGET)
	@echo "Building persistent on-disk index..."
	@./$(TARGET) --build-index

benchmark-indexing: $(TARGET)
	@echo "Running indexing scalability benchmark..."
	@./$(TARGET) --benchmark-indexing

$(TARGET): $(ALL_OBJS)
	@echo "[LINK] Building executable: $@"
	@$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@
	@echo "Build complete: $@"

# --- Main Benchmark Target ---
benchmark: $(TARGET) benchmark-all plot

benchmark-all:
	@echo "Running comprehensive scalability benchmarks..."
	@echo "Results will be logged to $(LOG_FILE)"
	@rm -f $(LOG_FILE) $(CSV_FILE)
	@touch $(LOG_FILE)
	@# Run Boolean benchmarks with varying CPU workers
	@echo "\n--- Running Boolean Scalability Tests ---" | tee -a $(LOG_FILE)
	@for workers in $(CPU_WORKER_COUNTS); do \
		echo "\n[BENCHMARK] Boolean with $$workers CPU workers..." | tee -a $(LOG_FILE); \
		CILK_NWORKERS=$$workers ./$(TARGET) --benchmark --label "Boolean_$$workers-cpu" --cpu-workers $$workers --no-rerank | tee -a $(LOG_FILE); \
	done
	@# Run Reranking benchmarks with varying CPU workers
	@echo "\n--- Running Reranking Scalability Tests ---" | tee -a $(LOG_FILE)
	@for cpu_w in $(CPU_WORKER_COUNTS); do \
		echo "\n[BENCHMARK] Reranking with $$cpu_w CPU workers..." | tee -a $(LOG_FILE); \
		CILK_NWORKERS=$$cpu_w ./$(TARGET) --benchmark --label "Rerank_$$cpu_w-cpu" --cpu-workers $$cpu_w | tee -a $(LOG_FILE); \
	done
	@echo "\nAll benchmarks completed. Consolidated results in $(CSV_FILE)"

plot:
	@echo "Generating performance plots from benchmark results..."
	python3 scripts/evaluation_metrics.py --results $(CSV_FILE)

run: $(TARGET)
	@echo "Running quick demo..."
	@./$(TARGET) --demo

# --- Object File Compilation Rules ---
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@echo "[CXX] Compiling $<"
	@$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	@echo "Cleaning build"
	@rm -rf build
	@echo "Clean complete."

clean-all:
	@echo "Cleaning build, models, results, index, data"
	@rm -rf build $(INDEX_DIR) $(RESULTS_DIR)/*.log $(RESULTS_DIR)/*.csv $(RESULTS_DIR)/*.png $(DATA_DIR) $(MODEL_DIR) 
	@echo "Clean complete."
