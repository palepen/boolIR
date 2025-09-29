# --- Compiler and Flags ---
CXX = /opt/opencilk/bin/clang++
CXXFLAGS = -std=c++17 -fopencilk -O3 -pthread -Iinclude
LDFLAGS = -L/opt/opencilk/lib

# --- Executable Targets ---
APP_TARGET = indexing_benchmark
TEST_TARGET = test_postings

# --- Source Files ---
APP_SRCS = src/benchmark_indexing.cpp src/indexing/parallel_indexer.cpp src/indexing/performance_monitor.cpp src/indexing/posting_list.cpp src/indexing/sequential_indexer.cpp
TEST_SRCS = tests/test_posting_list.cpp src/indexing/posting_list.cpp

# --- Build Rules ---
.PHONY: all test clean

all: $(APP_TARGET)

$(APP_TARGET): $(APP_SRCS)
	@echo "Building application: $@"
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@
	@echo "Build complete."

$(TEST_TARGET): $(TEST_SRCS)
	@echo "Building test: $@"
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@
	@echo "Test build complete."

test: $(TEST_TARGET)
	@echo "--- Running Unit Test ---"
	./$(TEST_TARGET)
	@echo "-----------------------"

clean:
	@echo "Cleaning up..."
	rm -f $(APP_TARGET) $(TEST_TARGET)
