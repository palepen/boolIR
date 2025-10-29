OPENCILK_PATH := /opt/opencilk/
LIBTORCH_PATH := /opt/libtorch
CUDA_INCLUDE_PATH := /usr/local/cuda-13/include

CUDA_LIB_PATH := /usr/local/cuda-13.0/targets/x86_64-linux/lib

CXX = $(OPENCILK_PATH)/bin/clang++
CPU_WORKER_COUNTS = 1 2 4 8 12

CXXFLAGS = -std=c++17 -fopencilk -O3 -pthread \
           -Iinclude \
           -I$(LIBTORCH_PATH)/include \
           -I$(LIBTORCH_PATH)/include/torch/csrc/api/include \
           -I$(CUDA_INCLUDE_PATH) \
		   -MMD -MP

LDFLAGS = -L$(OPENCILK_PATH)/lib \
          -L$(LIBTORCH_PATH)/lib \
          -L$(CUDA_LIB_PATH) \
      	  -no-pie \
          -Wl,-rpath,$(OPENCILK_PATH)/lib \
          -Wl,-rpath,$(LIBTORCH_PATH)/lib \
          -Wl,-rpath,$(CUDA_LIB_PATH) \
          -Wl,--no-as-needed -ltorch -ltorch_cpu -ltorch_cuda -lc10 -lc10_cuda -Wl,--as-needed \
          -lcudart


# --- Directories ---
SRC_DIR = src
OBJ_DIR = build/obj
BIN_DIR = build/bin
RESULTS_DIR = results
INDEX_DIR = index

# --- Target Executable ---
TARGET = $(BIN_DIR)/full_system_benchmark

# --- Source Files ---
CORE_SRCS = $(SRC_DIR)/main.cpp \
         	$(SRC_DIR)/system_controller.cpp \
            $(SRC_DIR)/benchmark_suite.cpp \
            $(SRC_DIR)/data_loader.cpp \
            $(SRC_DIR)/document_store.cpp

COMMON_SRCS = $(SRC_DIR)/common/utils.cpp 

INDEX_SRCS = $(SRC_DIR)/indexing/bsbi_indexer.cpp \
             $(SRC_DIR)/indexing/posting_list.cpp \
             $(SRC_DIR)/indexing/performance_monitor.cpp

RETRIEVAL_SRCS = $(SRC_DIR)/retrieval/retrieval_set.cpp \
                 $(SRC_DIR)/retrieval/retriever.cpp \
       			 $(SRC_DIR)/retrieval/query_expander.cpp \
                 $(SRC_DIR)/retrieval/query_preprocessor.cpp

RERANK_SRCS = $(SRC_DIR)/reranking/neural_reranker.cpp
TOKEN_SRCS = $(SRC_DIR)/tokenizer/wordpiece_tokenizer.cpp
EVAL_SRCS = $(SRC_DIR)/evaluation/evaluator.cpp

ALL_SRCS = $(CORE_SRCS) $(COMMON_SRCS) $(INDEX_SRCS) $(RETRIEVAL_SRCS) $(RERANK_SRCS) $(TOKEN_SRCS) $(EVAL_SRCS) 
ALL_OBJS = $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(ALL_SRCS))

DEPS = $(ALL_OBJS:.o=.d)

.PHONY: all clean model dirs index run benchmark benchmark-indexing plot

all: dirs $(TARGET)

dirs:
	@mkdir -p $(OBJ_DIR)/indexing $(OBJ_DIR)/retrieval $(OBJ_DIR)/reranking $(OBJ_DIR)/tokenizer $(OBJ_DIR)/evaluation $(OBJ_DIR)/common # <-- ADDED $(OBJ_DIR)/common
	@mkdir -p $(BIN_DIR) $(RESULTS_DIR) $(INDEX_DIR)

model:
	@echo "Exporting BERT cross-encoder model to TorchScript format..."
	@python3 scripts/export_model.py

index: $(TARGET)
	@echo "Building sharded on-disk index (64 shards)..."
	CILK_NWORKERS=6 ./$(TARGET) --build-index --shards 64

$(TARGET): $(ALL_OBJS)
	@echo "[LINK] Building executable: $@"
	@$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@
	@echo "Build complete: $@"

# --- Main Benchmark & Run Targets ---
benchmark-indexing: $(TARGET)
	@echo "Running indexing scalability benchmark..."
	CILK_NWORKERS=6 ./$(TARGET) --benchmark-indexing

# --- Main Benchmark & Run Targets (Simplified) ---
benchmark-workers: $(TARGET)
	@echo "Running integrated query performance benchmarks..."
	@for workers in $(CPU_WORKER_COUNTS); do \
		echo "\n[BENCHMARK] Running with $${workers} CPU workers..." | tee -a $(LOG_FILE); \
		CILK_NWORKERS=$${workers} ./$(TARGET) --benchmark --label "Sharded_$${workers}-cpu" --cpu-workers $${workers} | tee -a $(LOG_FILE); \
	done
	@echo "\nAll benchmarks completed. Consolidated results in $(CSV_FILE)"


benchmark: $(TARGET)
	CILK_NWORKERS=6 ./$(TARGET) --benchmark --label "worker-6-cpu" --cpu-workers 6 | tee -a $(LOG_FILE); 
	

run: $(TARGET)
	@echo "Running interactive mode..."
	@./$(TARGET) --interactive

dataset:
	@echo "Running Dataset Fetching Script"
	python3 scripts/download_dataset.py

plot: $(RESULTS_DIR)/all_benchmarks.csv
	python3 scripts/evaluation_metrics.py --results $(RESULTS_DIR)/all_benchmarks.csv

# --- Object File Compilation Rules ---
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@echo "[CXX] Compiling $<"
	@$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	@echo "Cleaning build artifacts"
	@rm -rf build
	@echo "Clean complete."

-include $(DEPS)