# --- Compiler and Flags ---
CXX = /opt/opencilk/bin/clang++
CXXFLAGS = -std=c++17 -fopencilk -O3 -pthread -Iinclude -I/opt/onnxruntime/include
LDFLAGS = -L/opt/opencilk/lib -L/opt/onnxruntime/lib -lonnxruntime \
          -Wl,-rpath,/opt/opencilk/lib,-rpath,/opt/onnxruntime/lib

# --- Executable Target ---
TARGET = full_system_benchmark

# --- Source Files ---
SRCS = src/main.cpp \
       src/system_controller.cpp \
       src/indexing/parallel_indexer.cpp \
       src/indexing/posting_list.cpp \
       src/indexing/performance_monitor.cpp \
       src/retrieval/parallel_retrieval.cpp \
       src/retrieval/result_set.cpp \
       src/reranking/neural_reranker.cpp

# --- Build Rules ---
.PHONY: all clean model

all: $(TARGET)

# Rule to export the BERT model to ONNX format
model:
	@echo "Exporting BERT model to ONNX format..."
	python3 scripts/export_model.py

$(TARGET): $(SRCS)
	@echo "Building full system application: $@"
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@
	@echo "Build complete."

clean:
	@echo "Cleaning up..."
	rm -f $(TARGET)