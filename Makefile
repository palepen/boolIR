# OpenCilk C++ Compiler and flags
CXX = clang++
CILK_CXX = /opt/opencilk/bin/clang++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2 -g
CILK_FLAGS = -fopencilk
INCLUDES = -Iinclude
LIBS = -lm -lpthread -lstdc++

# Directories
SRC_DIR = src
BUILD_DIR = build
DOCS_DIR = docs
MODELS_DIR = models
RESULTS_DIR = results
SCRIPTS_DIR = scripts
DATA_DIR = data

# Source files
SOURCES = $(wildcard $(SRC_DIR)/*.cpp)
OBJECTS = $(SOURCES:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)
TARGET = boolean_retrieval

# --- Python Environment ---
VENV_DIR = .venv
PYTHON = $(VENV_DIR)/bin/python3
PIP = $(VENV_DIR)/bin/pip

# Default target
all: setup $(TARGET)

# Setup C++ directories
setup:
	@mkdir -p $(BUILD_DIR) $(RESULTS_DIR) $(DOCS_DIR) $(MODELS_DIR)

# Build main C++ target
$(TARGET): $(OBJECTS)
	$(CILK_CXX) $(CXXFLAGS) $(CILK_FLAGS) $(OBJECTS) -o $(TARGET) $(LIBS)
	@echo "Build complete: $(TARGET)"

# Compile C++ source files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(BUILD_DIR)
	$(CILK_CXX) $(CXXFLAGS) $(CILK_FLAGS) $(INCLUDES) -c $< -o $@

# --- Python, Model, and Dataset Targets ---

# Create Python virtual environment and install dependencies
setup-python: $(VENV_DIR)/touchfile

$(VENV_DIR)/touchfile: $(SCRIPTS_DIR)/requirements.txt
	@echo "Creating Python virtual environment..."
	python3 -m venv $(VENV_DIR)
	@echo "Installing Python dependencies..."
	$(PIP) install -r $(SCRIPTS_DIR)/requirements.txt
	touch $(VENV_DIR)/touchfile

# Download and export the BERT model
fetch-model: setup-python
	@echo "Fetching and exporting BERT model..."
	$(PYTHON) $(SCRIPTS_DIR)/export_model.py

# Download and prepare the dataset
dataset:
	@echo "Fetching and preparing dataset..."
	@bash $(SCRIPTS_DIR)/download_trec6.sh

# Install all dependencies (OpenCilk + Python + Dataset)
install-deps: setup-python
	@echo "\nPython dependencies installed."
	@echo "Please ensure OpenCilk is installed separately."
	@echo "Official guide: https://www.opencilk.org/doc/users-guide/install/"

# --- Utility Targets ---

# Clean build files
clean:
	rm -rf $(BUILD_DIR) $(TARGET) *_cilksan *_cilkscale

# Deep clean (including results, Python venv, and data)
distclean: clean
	rm -rf $(RESULTS_DIR)/* $(VENV_DIR) $(MODELS_DIR)/*.onnx $(DOCS_DIR)/*

# Run interactive mode
run-interactive: all
	@echo "Starting interactive mode..."
	./$(TARGET) --mode interactive --dataset $(DOCS_DIR)

# Run evaluation mode
# NOTE: You must update the paths to your query and qrels files below.
run-evaluation: all
	@echo "Starting evaluation mode..."
	./boolean_retrieval --mode evaluation \
		--dataset $(DOCS_DIR) \
		--queries $(DATA_DIR)/topics.301-350.txt \
		--qrels $(DATA_DIR)/qrels.trec6.adhoc.txt

# --- Full Pipeline ---
full-setup: install-deps dataset fetch-model
	@echo "✅ Full project setup is complete."
full-run: all
	@echo "Project built. Run with './boolean_retrieval --mode interactive --dataset docs'"

# Help target
help:
	@echo "Available targets:"
	@echo "  all             - Build the C++ project"
	@echo "  setup-python    - Create Python venv and install dependencies"
	@echo "  fetch-model     - Download and export the BERT model"
	@echo "  dataset         - Download and prepare the dataset"
	@echo "  install-deps    - Set up all Python dependencies"
	@echo "  full-setup      - Run the complete setup (Python, Dataset, Model)"
	@echo "  run-interactive - Build and run the interactive mode"
	@echo "  run-evaluation  - Run relevance evaluation against a ground truth"
	@echo "  clean           - Remove build files"
	@echo "  distclean       - Deep clean the project"
	@echo "  help            - Show this help message"

.PHONY: all setup clean distclean fetch-model install-deps setup-python dataset run-interactive run-evaluation full-setup full-run help