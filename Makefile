# --- Compiler and Flags ---
CXX = /opt/opencilk/bin/clang++
CXXFLAGS = -std=c++17 -fopencilk -O3 -pthread -Iinclude -I/opt/onnxruntime/include
LDFLAGS = -L/opt/opencilk/lib -L/opt/onnxruntime/lib -lonnxruntime \
          -Wl,-rpath,/opt/opencilk/lib,-rpath,/opt/onnxruntime/lib

# --- Directories ---
BUILD_DIR = build
SRC_DIR = src
OBJ_DIR = $(BUILD_DIR)/obj
BIN_DIR = $(BUILD_DIR)/bin
INCLUDE_DIR = include
RESULTS_DIR = results

# --- Target Executable ---
TARGET = $(BIN_DIR)/full_system_benchmark

# --- Source Files ---
CORE_SRCS = $(SRC_DIR)/main.cpp \
            $(SRC_DIR)/system_controller.cpp \
            $(SRC_DIR)/benchmark_suite.cpp \
            $(SRC_DIR)/data_loader.cpp

INDEX_SRCS = $(SRC_DIR)/indexing/parallel_indexer.cpp \
             $(SRC_DIR)/indexing/posting_list.cpp \
             $(SRC_DIR)/indexing/performance_monitor.cpp

RETRIEVAL_SRCS = $(SRC_DIR)/retrieval/retrieval_set.cpp \
                 $(SRC_DIR)/retrieval/optimized_parallel_retrieval.cpp

RERANK_SRCS = $(SRC_DIR)/reranking/neural_reranker.cpp \
              $(SRC_DIR)/reranking/parallel_gpu_reranking.cpp

TOKEN_SRCS = $(SRC_DIR)/tokenizer/wordpiece_tokenizer.cpp

EVAL_SRCS = $(SRC_DIR)/evaluation/evaluator.cpp

ALL_SRCS = $(CORE_SRCS) $(INDEX_SRCS) $(RETRIEVAL_SRCS) $(RERANK_SRCS) $(TOKEN_SRCS) $(EVAL_SRCS)

# --- Object Files ---
ALL_OBJS = $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(ALL_SRCS))

# --- Build Rules ---
.PHONY: all clean model dirs run benchmark debug profile help

all: dirs $(TARGET)

help:
	@echo "========================================="
	@echo "High-Performance IR System Build Options"
	@echo "========================================="
	@echo "make          - Build the system"
	@echo "make clean    - Remove all built files"
	@echo "make model    - Export BERT model to ONNX"
	@echo "make run      - Build and run quick demo"
	@echo "make benchmark - Build and run full benchmarks"
	@echo "make debug    - Build with debug symbols"
	@echo "make profile  - Build with profiling enabled"
	@echo "========================================="

dirs:
	@mkdir -p $(BUILD_DIR)
	@mkdir -p $(OBJ_DIR)/indexing
	@mkdir -p $(OBJ_DIR)/retrieval
	@mkdir -p $(OBJ_DIR)/reranking
	@mkdir -p $(OBJ_DIR)/tokenizer
	@mkdir -p $(OBJ_DIR)/evaluation
	@mkdir -p $(BIN_DIR)
	@mkdir -p $(RESULTS_DIR)

model:
	@echo "Exporting BERT cross-encoder model to ONNX format..."
	python3 scripts/export_model.py

$(TARGET): $(ALL_OBJS)
	@echo "[LINK] Building executable: $@"
	@$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@
	@echo "Build complete: $@"

# Generic rule for compiling .cpp files to .o files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@echo "[CC] $<"
	@$(CXX) $(CXXFLAGS) -c $< -o $@

# Run targets
run: $(TARGET)
	@echo "Running quick demo..."
	@./$(TARGET) --demo

benchmark: $(TARGET)
	@echo "Running comprehensive benchmarks..."
	@./$(TARGET) --benchmark

# Debug build
debug: CXXFLAGS += -g -DDEBUG -fsanitize=address
debug: LDFLAGS += -fsanitize=address
debug: clean all

# Profile build
profile: CXXFLAGS += -pg -g
profile: LDFLAGS += -pg
profile: clean all

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	@rm -rf $(BUILD_DIR)
	@echo "Clean complete."