# --- Compiler and Flags ---
CXX = /opt/opencilk/bin/clang++
CXXFLAGS = -std=c++17 -fopencilk -O3 -pthread -Iinclude -I/opt/onnxruntime/include
# **THE FIX IS HERE**: Added -no-pie to the linker flags.
LDFLAGS = -L/opt/opencilk/lib -L/opt/onnxruntime/lib -lonnxruntime -no-pie -lstemmer \
          -Wl,-rpath,/opt/opencilk/lib,-rpath,/opt/onnxruntime/lib 

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
            $(SRC_DIR)/data_loader.cpp

INDEX_SRCS = $(SRC_DIR)/indexing/bsbi_indexer.cpp \
             $(SRC_DIR)/indexing/posting_list.cpp \
             $(SRC_DIR)/indexing/performance_monitor.cpp

RETRIEVAL_SRCS = $(SRC_DIR)/retrieval/retrieval_set.cpp \
                 $(SRC_DIR)/retrieval/optimized_parallel_retrieval.cpp \
                 $(SRC_DIR)/retrieval/query_expander.cpp

RERANK_SRCS = $(SRC_DIR)/reranking/neural_reranker.cpp \
              $(SRC_DIR)/reranking/parallel_gpu_reranking.cpp \

TOKEN_SRCS = $(SRC_DIR)/tokenizer/wordpiece_tokenizer.cpp \
             $(SRC_DIR)/tokenizer/porter_stemmer.cpp

EVAL_SRCS = $(SRC_DIR)/evaluation/evaluator.cpp

ALL_SRCS = $(CORE_SRCS) $(INDEX_SRCS) $(RETRIEVAL_SRCS) $(RERANK_SRCS) $(TOKEN_SRCS) $(EVAL_SRCS)
ALL_OBJS = $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(ALL_SRCS))

.PHONY: all clean model dirs index run benchmark test-stemmer compare-stemming

all: dirs $(TARGET)

# Test stemmer
test-stemmer: dirs $(OBJ_DIR)/tokenizer/porter_stemmer.o
	@echo "Building stemmer test..."
	@$(CXX) $(CXXFLAGS) test_stemmer.cpp $(OBJ_DIR)/tokenizer/porter_stemmer.o $(LDFLAGS) -o $(BIN_DIR)/test_stemmer
	@echo "Running stemmer test..."
	@./$(BIN_DIR)/test_stemmer

# Compare vocabulary with/without stemming
compare-stemming: dirs $(OBJ_DIR)/tokenizer/porter_stemmer.o
	@echo "Building vocabulary comparison tool..."
	@$(CXX) $(CXXFLAGS) compare_stemming.cpp $(OBJ_DIR)/tokenizer/porter_stemmer.o $(LDFLAGS) -o $(BIN_DIR)/compare_stemming
	@echo "Analyzing first batch file..."
	@./$(BIN_DIR)/compare_stemming data/cord19-trec-covid_corpus_batched/batch_00000.txt

dirs:
	@mkdir -p $(OBJ_DIR)/indexing
	@mkdir -p $(OBJ_DIR)/retrieval
	@mkdir -p $(OBJ_DIR)/reranking
	@mkdir -p $(OBJ_DIR)/tokenizer
	@mkdir -p $(OBJ_DIR)/evaluation
	@mkdir -p $(BIN_DIR)
	@mkdir -p $(RESULTS_DIR)
	@mkdir -p $(INDEX_DIR)

model:
	@echo "Exporting BERT cross-encoder model to ONNX format..."
	python3 scripts/export_model.py

index: $(TARGET)
	@echo "Building persistent on-disk index with Porter stemming..."
	@./$(TARGET) --build-index

$(TARGET): $(ALL_OBJS)
	@echo "[LINK] Building executable: $@"
	@$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@
	@echo "Build complete: $@"

# --- Explicit Compilation Rules ---
$(OBJ_DIR)/main.o: $(SRC_DIR)/main.cpp
	@$(CXX) $(CXXFLAGS) -c $< -o $@
$(OBJ_DIR)/system_controller.o: $(SRC_DIR)/system_controller.cpp
	@$(CXX) $(CXXFLAGS) -c $< -o $@
$(OBJ_DIR)/benchmark_suite.o: $(SRC_DIR)/benchmark_suite.cpp
	@$(CXX) $(CXXFLAGS) -c $< -o $@
$(OBJ_DIR)/data_loader.o: $(SRC_DIR)/data_loader.cpp
	@$(CXX) $(CXXFLAGS) -c $< -o $@
$(OBJ_DIR)/indexing/bsbi_indexer.o: $(SRC_DIR)/indexing/bsbi_indexer.cpp
	@$(CXX) $(CXXFLAGS) -c $< -o $@
$(OBJ_DIR)/indexing/posting_list.o: $(SRC_DIR)/indexing/posting_list.cpp
	@$(CXX) $(CXXFLAGS) -c $< -o $@
$(OBJ_DIR)/indexing/performance_monitor.o: $(SRC_DIR)/indexing/performance_monitor.cpp
	@$(CXX) $(CXXFLAGS) -c $< -o $@
$(OBJ_DIR)/retrieval/retrieval_set.o: $(SRC_DIR)/retrieval/retrieval_set.cpp
	@$(CXX) $(CXXFLAGS) -c $< -o $@
$(OBJ_DIR)/retrieval/optimized_parallel_retrieval.o: $(SRC_DIR)/retrieval/optimized_parallel_retrieval.cpp
	@$(CXX) $(CXXFLAGS) -c $< -o $@
$(OBJ_DIR)/retrieval/query_expander.o: $(SRC_DIR)/retrieval/query_expander.cpp
	@$(CXX) $(CXXFLAGS) -c $< -o $@
$(OBJ_DIR)/reranking/neural_reranker.o: $(SRC_DIR)/reranking/neural_reranker.cpp
	@$(CXX) $(CXXFLAGS) -c $< -o $@
$(OBJ_DIR)/reranking/parallel_gpu_reranking.o: $(SRC_DIR)/reranking/parallel_gpu_reranking.cpp
	@$(CXX) $(CXXFLAGS) -c $< -o $@
$(OBJ_DIR)/reranking/gpu_worker_pool.o: $(SRC_DIR)/reranking/gpu_worker_pool.cpp
	@$(CXX) $(CXXFLAGS) -c $< -o $@
$(OBJ_DIR)/tokenizer/wordpiece_tokenizer.o: $(SRC_DIR)/tokenizer/wordpiece_tokenizer.cpp
	@$(CXX) $(CXXFLAGS) -c $< -o $@
$(OBJ_DIR)/tokenizer/porter_stemmer.o: $(SRC_DIR)/tokenizer/porter_stemmer.cpp
	@$(CXX) $(CXXFLAGS) -c $< -o $@
$(OBJ_DIR)/evaluation/evaluator.o: $(SRC_DIR)/evaluation/evaluator.cpp
	@$(CXX) $(CXXFLAGS) -c $< -o $@

run: $(TARGET)
	@echo "Running quick demo..."
	@./$(TARGET) --demo

benchmark: $(TARGET)
	@echo "Running comprehensive benchmarks..."
	@./$(TARGET) --benchmark | tee $(RESULTS_DIR)/output.log

clean:
	@echo "Cleaning build artifacts..."
	@rm -rf build $(INDEX_DIR)
	@echo "Clean complete."