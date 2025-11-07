# Boolean Information Retrieval System

[![C++17](https://img.shields.io/badge/C++-17-blue.svg)](https://isocpp.org/)
[![OpenCilk](https://img.shields.io/badge/Parallel-OpenCilk-green.svg)](https://www.opencilk.org/)
[![LibTorch](https://img.shields.io/badge/ML-LibTorch-red.svg)](https://pytorch.org/)

A scalable, memory-efficient information retrieval system implementing **streaming BSBI indexing** and **neural reranking** for document search. Capable of indexing corpora larger than available RAM with constant memory usage.

## Key Features

- **Indexing**: Index corpora exceeding RAM capacity using streaming architecture
- **Parallel Processing**: OpenCilk-based parallelization for multi-core CPUs
- **Neural Reranking**: BERT-based cross-encoder for improved search quality
- **Memory Efficient**: Constant memory footprint regardless of corpus size
- **Sharded Architecture**: 64-shard distributed index for parallel query processing

## Quick Start

```bash
# Build the system
make clean && make all

# Download dataset (TREC-COVID)
make dataset

# Export neural model
make model

# Build index (streaming, memory-efficient)
make index

# Run performance tests
make all-benchmarks

# Interactive search
make run
```

---

## System Architecture

### High-Level Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                     IR SYSTEM ARCHITECTURE                          │
└─────────────────────────────────────────────────────────────────────┘

           ┌──────────────┐
           │   Documents  │
           │  (Disk/SSD)  │
           └──────┬───────┘
                  │
                  ▼
        ┌─────────────────────┐
        │  INDEXING PIPELINE  │
        └─────────────────────┘
                  │
        ┌─────────┴─────────┐
        ▼                   ▼
┌───────────────┐     ┌──────────────────┐
│ Document      │     │                  │
│ Stream        │──▶ │ Indexer          │
│ (mmap I/O)    │     │ (Parallel)       │
└───────────────┘     └────────┬─────────┘
                               │
                ┌──────────────┴──────────┐
                ▼                         ▼
        ┌──────────────┐         ┌──────────────┐
        │ Inverted     │         │ Document     │
        │ Index        │         │ Store        │
        │ (64 shards)  │         │ (Compressed) │
        └──────┬───────┘         └──────┬───────┘
               │                        │
               └────────┬───────────────┘
                        ▼
              ┌──────────────────┐
              │  QUERY PIPELINE  │
              └──────────────────┘
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
┌──────────────┐ ┌────────────┐ ┌─────────────────┐
│ Query        │ │ Boolean    │ │ Neural          │
│ Expansion    │─│ Retrieval  │─│ Reranker        │
│ (Synonyms)   │ │ (Parallel) │ │ (GPU/BERT)      │
└──────────────┘ └────────────┘ └─────────────────┘
                        │
                        ▼
                ┌──────────────┐
                │ Ranked       │
                │ Results      │
                └──────────────┘
```

### Component Breakdown

#### 1. Streaming Indexing Layer

- **DocumentStream**: Memory-mapped file reader for on-demand document access
- **Indexer**: Blocked sort-based indexing with constant memory usage
- **Parallel Run Generation**: Document-level partitioning across workers
- **External Merge Sort**: Multi-pass parallel merging of sorted runs

#### 2. Memory-Mapped I/O

- Zero-copy document reading using `mmap()`
- OS-managed page cache for frequently accessed documents
- Sequential access pattern optimization with `MADV_SEQUENTIAL`
- Thread-safe read-only mappings for parallel access

#### 3. Parallel Retrieval Engine

- **Sharded Index**: 64 shards for load distribution
- **Boolean Query Processing**: AND, OR, NOT operators with query trees
- **Query Expansion**: Synonym-based term expansion
- **Set Operations**: Optimized intersection/union of posting lists

#### 4. Neural Reranking

- **BERT Cross-Encoder**: Pre-trained model fine-tuned for relevance
- **Batch Processing**: GPU-accelerated scoring of top-K candidates
- **Chunked Inference**: Memory-efficient processing of large candidate sets
- **LibTorch Integration**: C++ inference with TorchScript models

---

## Detailed Architecture

### Indexing Architecture

#### Indexing Algorithm

The system implements a **streaming variant** of BSBI that processes documents on-demand rather than loading them into memory:

```
Traditional BSBI:
┌─────────────────┐
│ Load ALL docs   │  ← Memory = Corpus Size (FAILS for large corpora)
│ into RAM        │
└────────┬────────┘
         ▼
┌─────────────────┐
│ Process blocks  │
└─────────────────┘

Streaming BSBI:
┌─────────────────┐
│ Build metadata  │  ← Memory = O(num_docs × metadata_size)
│ index only      │     ~ 50 MB for 1M documents
└────────┬────────┘
         ▼
┌─────────────────┐
│ Stream docs     │  ← Memory = O(block_size)
│ one-by-one      │     Constant, configurable
└─────────────────┘
```

#### Phase 1: Generate Sorted Runs

**Objective**: Convert documents into sorted (term, doc_id) pairs in manageable blocks.

```
Worker 0                Worker 1                Worker N
(docs 0-999)           (docs 1000-1999)        (docs N*1000-...)
     │                       │                        │
     ├─ Stream doc 0         ├─ Stream doc 1000      ├─ Stream doc N*1000
     │  (mmap read)          │                        │
     ├─ Tokenize             ├─ Tokenize             ├─ Tokenize
     │  (preprocessing)      │                        │
     ├─ Add to buffer        ├─ Add to buffer        ├─ Add to buffer
     │  [(term,docid),...]   │                        │
     │                       │                        │
     ├─ Buffer full?         ├─ Buffer full?         ├─ Buffer full?
     │  YES: Sort & Write    │  YES: Sort & Write    │  YES: Sort & Write
     │       run_w0_b0.dat   │       run_w1_b0.dat   │       run_wN_b0.dat
     │       Clear buffer    │       Clear buffer    │       Clear buffer
     │                       │                        │
     └─ Next doc...          └─ Next doc...          └─ Next doc...

Memory per worker = block_size_bytes (e.g., 256 MB)
Total memory = num_workers × block_size_bytes
```

**Parallelization Strategy**: Document-level partitioning

- Each worker processes a disjoint range of document IDs
- Independent run file generation → no synchronization needed

**Memory Management**:

```
Block Buffer Size = 256 MB (configurable)
├─ Term-doc pairs: ~240 MB
├─ Processing overhead: ~10 MB
└─ Document content: ~6 MB (transient, freed after processing)

Total System Memory = num_workers × block_size + overhead
Example: 12 workers × 256 MB + 100 MB = ~3.2 GB
```

#### Phase 2: Merge Sorted Runs

**Objective**: Merge multiple sorted run files into a single sorted file.

```
Pass 1: Pairwise Merge
┌────────┐  ┌────────┐     ┌────────┐  ┌────────┐
│ run_0  │──│ run_1  │     │ run_2  │──│ run_3  │
└────┬───┘  └───┬────┘     └────┬───┘  └───┬────┘
     └──────────┴──▶ merge       └──────────┴──▶ merge
            │                           │
      ┌─────▼─────┐               ┌─────▼─────┐
      │ merged_0  │               │ merged_1  │
      └───────────┘               └───────────┘

Pass 2: Continue merging...
      ┌───────────┐               ┌───────────┐
      │ merged_0  │───────────────│ merged_1  │
      └─────┬─────┘               └─────┬─────┘
            └───────────┬───────────────┘
                        ▼
                  ┌───────────┐
                  │   final   │
                  │    run    │
                  └───────────┘
```

**Algorithm**: Two-way external merge

```
merge(file1, file2, output):
    pair1 = read_next(file1)
    pair2 = read_next(file2)

    while both files have data:
        if pair1 < pair2:
            write(output, pair1)
            pair1 = read_next(file1)
        else:
            write(output, pair2)
            pair2 = read_next(file2)

    flush_remaining(file1, output)
    flush_remaining(file2, output)
```

**Parallel Execution**: All pairs in a pass merge simultaneously using `cilk_for`.

**Memory Usage**: O(1) - only two pairs in memory at once per merge operation.

#### Phase 3: Create Sharded Index

**Objective**: Convert sorted run into dictionary + posting lists, distributed across shards.

```
Final Sorted Run:
┌──────────────────────────────────────────────────────┐
│ apple,1  apple,5  banana,2  cat,3  cat,8  dog,6  ... │
└──────────────────────────────────────────────────────┘
                      │
                      ▼
            ┌─────────────────┐
            │ Hash each term  │
            │ to determine    │
            │ shard           │
            └────────┬────────┘
                     │
        ┌────────────┼────────────┐
        ▼            ▼            ▼
   ┌─────────┐  ┌─────────┐  ┌─────────┐
   │ Shard 0 │  │ Shard 1 │  │Shard 63 │
   └─────────┘  └─────────┘  └─────────┘

Each Shard contains:
┌─────────────────────┐
│ dict.dat            │  ← Term → (offset, size)
├─────────────────────┤
│ postings.dat        │  ← [doc_id, doc_id, ...]
└─────────────────────┘
```

**Sharding Strategy**:

- Hash-based distribution: `shard_id = hash(term) % 64`
- Load balancing: Hash function distributes terms uniformly
- Parallel queries: Different terms retrieved from different shards

**File Formats**:

```
Dictionary (dict.dat):
[term\0][offset:8bytes][size:8bytes][term\0][offset:8bytes][size:8bytes]...

Posting Lists (postings.dat):
[doc_id:4bytes][doc_id:4bytes][doc_id:4bytes]...
```

#### Phase 4: Create Document Store

**Objective**: Store preprocessed document content for reranking and display.

```
Document Store Structure:
┌─────────────────────────────────────────┐
│ documents.dat                           │
│ [id][length][content][id][length]...    │
└─────────────────────────────────────────┘
┌─────────────────────────────────────────┐
│ doc_offsets.dat                         │
│ [id][offset][id][offset][id][offset]... │
└─────────────────────────────────────────┘
┌─────────────────────────────────────────┐
│ doc_names.dat                           │
│ [id][name_len][name][id][name_len]...   │
└─────────────────────────────────────────┘
```

**Loading Strategy**: All documents loaded into memory for fast retrieval during reranking.

**Alternative** (future): On-demand loading for very large corpora.

---

### Query Processing Architecture

#### Query Flow

```
User Query: "coronavirus treatment"
         │
         ▼
┌──────────────────────┐
│ Query Preprocessing  │
│ - Lowercase          │
│ - Remove punctuation │
│ - Remove stopwords   │
└──────────┬───────────┘
           │ "coronavirus treatment"
           ▼
┌──────────────────────┐
│ Query Expansion      │
│ - Load synonyms      │
│ - Expand terms       │
│ - Build query tree   │
└──────────┬───────────┘
           │
           │ Query Tree:
           │     OR
           │    /  \
           │  covid  treatment
           │  /  \       /   \
           │ coronavirus therapy
           │ covid-19    ...
           ▼
┌──────────────────────┐
│ Boolean Retrieval    │
│ - Execute query tree │
│ - Parallel shard     │
│   lookups            │
│ - Set operations     │
└──────────┬───────────┘
           │ 1,234 candidate documents
           ▼
┌──────────────────────┐
│ Retrieve Top-K       │
│ (4,096 candidates)   │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Neural Reranking     │
│ - Load doc content   │
│ - BERT encoding      │
│ - Score pairs        │
│ - Sort by score      │
└──────────┬───────────┘
           │
           ▼
    ┌─────────────┐
    │ Top 10      │
    │ Results     │
    └─────────────┘
```

#### Boolean Retrieval

**Query Tree Execution**:

```
Execute AND:
    result = intersect(left_child, right_child)

Execute OR:
    result = union(left_child, right_child)

Execute NOT:
    result = difference(universe, child)

Execute TERM:
    shard_id = hash(term) % 64
    posting_list = load_postings(shard_id, term)
    return posting_list
```

**Parallel Shard Access**:

```
Query: "covid AND vaccine"

Thread 1:                    Thread 2:
├─ hash("covid") = 23       ├─ hash("vaccine") = 47
├─ load shard 23            ├─ load shard 47
├─ get postings for "covid" ├─ get postings for "vaccine"
└─ return [1,3,7,9,...]     └─ return [1,2,7,8,...]
         │                           │
         └───────────┬───────────────┘
                     ▼
              intersect([1,3,7,9], [1,2,7,8])
                     │
                     ▼
                [1, 7]  (result)
```

**Set Operations**: Optimized for sorted arrays

- Intersection: Two-pointer merge (O(n + m))
- Union: Two-pointer merge with deduplication
- Difference: Skip elements in second set

#### Neural Reranking

**Pipeline**:

```
Input: Query + Top-K documents (e.g., 4096)
       │
       ▼
Split into batches (batch_size = 128)
       │
       ├─ Batch 1 (docs 0-127)
       ├─ Batch 2 (docs 128-255)
       ├─ ...
       └─ Batch 32 (docs 3968-4095)
       │
       ▼
For each batch (GPU parallel):
    ├─ Tokenize: query + document → input_ids
    ├─ Encode: BERT(input_ids) → embeddings
    ├─ Score: Linear(embeddings) → relevance_score
    └─ Store: (doc_id, score)
       │
       ▼
Sort all batches by score (descending)
       │
       ▼
Return top 10
```

**BERT Cross-Encoder**:

```
Input Format:
[CLS] query tokens [SEP] document tokens [SEP] [PAD] [PAD] ...
  ^                  ^                      ^
  │                  │                      │
Special token   Separator          Separator for pair

Model Output:
Single relevance score per (query, document) pair
```

**Memory Management**:

- Pre-allocated GPU tensors for batch processing
- Reuse tensors across batches to avoid allocation overhead
- CPU-side document truncation to 256 words before GPU encoding

---

### Storage Architecture

#### Index Structure

```
index/
├── shard_0/
│   ├── dict.dat        ← Terms → (offset, size)
│   └── postings.dat    ← Sorted doc_id arrays
├── shard_1/
│   ├── dict.dat
│   └── postings.dat
├── ...
├── shard_63/
│   ├── dict.dat
│   └── postings.dat
├── documents.dat       ← Preprocessed document content
├── doc_offsets.dat     ← doc_id → file offset mapping
└── doc_names.dat       ← doc_id → filename mapping
```

#### Shard Distribution

**Hash Function**: `std::hash<std::string>{}(term) % 64`

**Example Distribution**:

```
Term        Hash    Shard
─────────────────────────
apple       12345   25
banana      67890   26
covid       23456   40
dog         45678   14
vaccine     89012   28
```

**Load Balancing**: Hash function ensures uniform distribution across shards.

**Query Benefit**: Multi-term queries access different shards in parallel.

#### Memory Layout

**In-Memory (Query Time)**:

```
┌─────────────────────────────────────────┐
│ Active Shard Dictionaries               │  ← Only for terms in query
│ (hash_map<string, DiskLocation>)        │     ~30 MB per shard
├─────────────────────────────────────────┤
│ Document Store (all documents)          │  ← Full corpus in memory
│ (hash_map<uint, Document>)              │     ~corpus_size × 0.8
├─────────────────────────────────────────┤
│ BERT Model Weights                      │  ← GPU memory
│ (LibTorch)                              │     ~400 MB
└─────────────────────────────────────────┘
```

**Disk-Only (Build Time)**:

```
temp/
├── run_w0_b0.dat       ← Intermediate sorted runs
├── run_w0_b1.dat
├── ...
└── final_run.dat       ← Merged run (deleted after indexing)
```

---

## Installation & Setup

### Prerequisites

**Hardware Requirements**:

- CPU: Multi-core processor (4+ cores recommended)
- RAM: 8 GB minimum, 16 GB recommended
- GPU: NVIDIA GPU with CUDA support (for neural reranking)
- Disk: 10 GB free space for index and models

**Software Dependencies**:

```
- C++ Compiler: Clang 11+ with OpenCilk support
- CUDA Toolkit: 11.0+ (for GPU reranking)
- LibTorch: 1.13+ (C++ PyTorch)
- Python: 3.8+ (for dataset download and evaluation)
- Make: GNU Make 4.0+
```

### Step-by-Step Installation

#### 1. Install OpenCilk Compiler

```bash
# Download OpenCilk
wget https://github.com/OpenCilk/opencilk-project/releases/download/opencilk%2Fv2.0/OpenCilk-2.0.0-x86_64-Linux-Ubuntu-22.04.sh

# Install
bash OpenCilk-2.0.0-x86_64-Linux-Ubuntu-22.04.sh --prefix=/opt/opencilk

# Add to PATH
export PATH=/opt/opencilk/bin:$PATH
```

#### 2. Install LibTorch

```bash
# Download LibTorch (CPU+CUDA)
wget https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.0.0%2Bcu118.zip

# Extract
unzip libtorch-cxx11-abi-shared-with-deps-2.0.0+cu118.zip -d /opt/

# Set library path
export LD_LIBRARY_PATH=/opt/libtorch/lib:$LD_LIBRARY_PATH
```

#### 3. Install Python Dependencies

```bash
pip3 install pandas matplotlib seaborn numpy ir_datasets
```

#### 4. Clone and Build

```bash
# Clone repository
git clone git@github.com:palepen/boolIR.git
cd boolIR

# Build
make clean
make all

# Verify build
./build/bin/full_system_benchmark --help
```

### Dataset Download

```bash
make dataset

# This will create:
# - data/cord19-trec-covid_corpus/  (documents)
# - data/topics.cord19-trec-covid.txt  (queries)
# - data/qrels.cord19-trec-covid.txt  (relevance judgments)
```

### Model Preparation

```bash
# Export BERT model to TorchScript format
make model

# This will create:
# - models/bert_model.pt  (TorchScript model)
# - models/vocab.txt  (WordPiece vocabulary)
```

### Build Index

```bash
# Build index with default settings (12 workers, 64 shards)
make index

# Or with custom settings
CILK_NWORKERS=8 ./build/bin/full_system_benchmark \
    --build-index \
    --shards 64

# Expected output:
# - index/shard_0/ through index/shard_63/
# - index/documents.dat
# - index/doc_offsets.dat
# - index/doc_names.dat
```

---

## Usage Instructions

### Building Index

#### Basic Usage

```bash
# Default: 12 workers, 64 shards, 256 MB block size
make index
```

#### Advanced Options

```bash
# Custom worker count
CILK_NWORKERS=8 ./build/bin/full_system_benchmark --build-index

# Custom shard count
./build/bin/full_system_benchmark --build-index --shards 128

# Both
CILK_NWORKERS=16 ./build/bin/full_system_benchmark \
    --build-index \
    --shards 64
```

#### Progress Tracking

During indexing, you'll see:

```
Building document stream index from: data/cord19-trec-covid_corpus
  Indexed 10000 documents...
  Indexed 20000 documents...
  ...

Phase 1: Generating sorted runs (streaming from disk)...
  Loading documents: [=================>  ] 75.3% (129000/171332) 8546/s ETA: 5s

Phase 2: Merging runs...
  Merge Pass 1: 48 files -> 24 files
  Merge Pass 2: 24 files -> 12 files
  ...

Phase 3: Creating 64 index shards...
  Sharded index created successfully.

Phase 4: Creating document store...
  Loading documents: [====================] 100.0% (171332/171332) 9234/s
```

### Running Queries

#### Interactive Mode

```bash
# Start interactive search
make run

# Or with query tree expansion
make run-log
```

**Example Session**:

```
Enter query (or 'exit' to quit): coronavirus treatment

--- Top 5 Pure Boolean Results (Unranked) ---
  Found 1234 total documents.
  1. Document: ./data/corpus/ug7v899j.txt (ID: 5678)
  2. Document: ./data/corpus/02tnwd4m.txt (ID: 1234)
  ...

--- Top 5 Neurally Reranked Results ---
  1. Document: ./data/corpus/ylzawv7k.txt (ID: 9012, Score: 0.9234)
  2. Document: ./data/corpus/ug7v899j.txt (ID: 5678, Score: 0.8967)
  ...

Total query time: 45.23 ms

Enter query (or 'exit' to quit): exit
```

### Benchmarking

#### Run All Benchmarks

```bash
# Run complete benchmark suite
# - Indexing scalability (1, 2, 4, 8, 12 workers)
# - Query scalability (1, 2, 4, 8, 12 workers)
make all-benchmarks

# Results saved to:
# - results/indexing_benchmarks.csv
# - results/all_benchmarks.csv
```

#### Indexing Benchmarks Only

```bash
# Test indexing with different worker counts
make benchmark-indexing-workers

# Results in: results/indexing_benchmarks.csv
```

#### Query Benchmarks Only

```bash
# Test query processing with different worker counts
make benchmark-query-workers

# Results in: results/all_benchmarks.csv
```

#### Single Configuration Test

```bash
# Test specific worker count
CILK_NWORKERS=8 ./build/bin/full_system_benchmark \
    --benchmark \
    --label "test_8_workers" \
    --cpu-workers 8
```

---

## Configuration

### Config.h Parameters

Located in `include/config.h`, this file controls all system behavior.

#### File Paths

```cpp
// Input data
const std::string CORPUS_DIR = "data/cord19-trec-covid_corpus";
const std::string TOPICS_PATH = "data/topics.cord19-trec-covid.txt";
const std::string QRELS_PATH = "data/qrels.cord19-trec-covid.txt";
const std::string SYNONYM_PATH = "data/synonyms.txt";

// Models
const std::string MODEL_PATH = "models/bert_model.pt";
const std::string VOCAB_PATH = "models/vocab.txt";

// Output
const std::string INDEX_PATH = "index";
const std::string TEMP_PATH = "index/temp";
const std::string RESULTS_CSV_PATH = "results/all_benchmarks.csv";
```

**Customization**: Change these paths to use different datasets or models.

#### Indexing Parameters

```cpp
// Number of index shards (for parallel query processing)
constexpr size_t DEFAULT_NUM_SHARDS = 64;
```

**Tuning Guide**:

- **More shards (128, 256)**: Better query parallelism, slightly larger dictionary overhead
- **Fewer shards (32, 16)**: Less overhead, may bottleneck on multi-term queries
- **Recommendation**: 64 shards for most workloads, 128 for high-concurrency scenarios

```cpp
// Block size for BSBI run generation (in MB)
constexpr size_t DEFAULT_BLOCK_SIZE_MB = 128;
```


**Tuning Guide**:

- **Smaller blocks (64-128 MB)**: Lower memory, more run files, slower merge
- **Larger blocks (256-512 MB)**: Higher memory, fewer run files, faster merge

#### Reranking Parameters

```cpp
// Maximum candidates to rerank (top-K from Boolean retrieval)
constexpr size_t MAX_RERANK_CANDIDATES = 4096;
```

**Trade-off**:

- **Higher values (8192, 16384)**: Better recall, slower reranking
- **Lower values (1024, 2048)**: Faster reranking, may miss relevant docs
- **Recommendation**: 4096 for balanced performance/quality

```cpp
// Maximum sequence length for BERT (in tokens)
constexpr int64_t MAX_SEQ_LEN = 512;
```

**Effect**:

- **512 tokens**: Standard BERT limit, ~400 words of context
- **Lower values (256, 128)**: Faster inference, truncates long documents
- **Recommendation**: Keep at 512 for full document context

```cpp
// Number of words to keep from each document
constexpr size_t DOCUMENT_TRUNCATE_WORDS = 256;
```

**Truncation Strategy**:

- Documents truncated to first N words before tokenization
- Reduces GPU memory and speeds up encoding
- **Recommendation**: 256 words captures most relevant content from abstracts

```cpp
// Batch size for GPU inference
constexpr size_t BATCH_SIZE = 128;
```

**GPU Memory**:

```
Memory ≈ batch_size × MAX_SEQ_LEN × model_hidden_size × bytes_per_param

```

**Tuning Guide**:

- **Smaller batches (32-64)**: GPUs with < 4 GB VRAM
- **Standard batches (128)**: GPUs with 6-8 GB VRAM
- **Larger batches (256-512)**: GPUs with 12+ GB VRAM
- **Recommendation**: Maximize batch size without OOM errors for best throughput

```cpp
// Chunk size for splitting large candidate sets
constexpr size_t GPU_CHUNK_SIZE = 256;
```

**Purpose**: When reranking > GPU_CHUNK_SIZE candidates, split into chunks to avoid OOM.

**Example**: 4096 candidates with chunk size 256 → 16 sequential GPU batches.

**Tuning Guide**:

- Match to GPU memory capacity
- Higher values = fewer GPU kernel launches = better throughput
- **Recommendation**: 256 for most GPUs, increase if you have high VRAM

#### Performance Tuning Summary

**For Low Memory Systems (< 8 GB RAM)**:

```cpp
constexpr size_t DEFAULT_BLOCK_SIZE_MB = 64;
constexpr size_t MAX_RERANK_CANDIDATES = 2048;
constexpr size_t BATCH_SIZE = 64;
```

Run with: `CILK_NWORKERS=4`

**For Standard Systems (16 GB RAM, 6 GB GPU)**:

```cpp
constexpr size_t DEFAULT_BLOCK_SIZE_MB = 256;
constexpr size_t MAX_RERANK_CANDIDATES = 4096;
constexpr size_t BATCH_SIZE = 128;
```

Run with: `CILK_NWORKERS=8`

**For High-End Systems (32+ GB RAM, 12+ GB GPU)**:

```cpp
constexpr size_t DEFAULT_BLOCK_SIZE_MB = 512;
constexpr size_t MAX_RERANK_CANDIDATES = 8192;
constexpr size_t BATCH_SIZE = 256;
```

Run with: `CILK_NWORKERS=12`

#### Worker Count Recommendations

The `CILK_NWORKERS` environment variable controls CPU parallelism:

```bash
# Auto-detect (uses all cores)
make index

# Explicit worker count
CILK_NWORKERS=8 make index
```

**Guidelines**:

- **Indexing**: Use physical cores (not hyperthreads) for best efficiency

  - 4-core CPU: `CILK_NWORKERS=4`
  - 8-core CPU: `CILK_NWORKERS=8`
  - 12-core CPU: `CILK_NWORKERS=12`

- **Query Processing**: Leave 1-2 cores for OS and other processes
  - 8-core CPU: `CILK_NWORKERS=6`
  - 12-core CPU: `CILK_NWORKERS=10`

**Finding Optimal Count**:

```bash
# Run scaling benchmark
make benchmark-indexing-workers

# Check efficiency in results/indexing_benchmarks.csv
# Use worker count where efficiency > 70%
```

---

## Future Work

### 1. Fine-Tuned Models

**Current**: Pre-trained BERT cross-encoder (MS MARCO)

**Enhancement**: Domain-specific fine-tuning

#### Strategy A: Fine-tune on TREC-COVID

```
Training Data: TREC-COVID qrels (relevance judgments)
  - 50 queries
  - ~50,000 query-document pairs
  - Binary labels (relevant/not relevant)

Training Process:
  1. Use existing BERT checkpoint
  2. Binary cross-entropy loss
  3. 3-5 epochs with learning rate 2e-5
  4. Validation on held-out queries

```

#### Strategy B: Contrastive Learning

```
Triplet Loss Training:
  - Anchor: Query
  - Positive: Relevant document
  - Negative: Non-relevant document (hard negatives)

Architecture:
  Query Encoder (BERT) ──┐
                          ├──> Similarity Score
  Document Encoder (BERT)─┘

Benefit: Faster inference (embed once, reuse for multiple queries)
```

### 2. Positional Index

**Current**: Inverted index with document IDs only

**Enhancement**: Store term positions for phrase queries and proximity scoring

#### Data Structure

```cpp
// Current
struct PostingList {
    std::vector<unsigned int> doc_ids;  // [1, 5, 9, 12, ...]
};

// Positional
struct PositionalPosting {
    unsigned int doc_id;
    std::vector<unsigned int> positions;  // [3, 15, 42, ...] (word offsets)
};

struct PositionalPostingList {
    std::vector<PositionalPosting> postings;
};
```

#### Use Cases

**Phrase Queries**:

```
Query: "machine learning"

Without positions:
  - Find docs with "machine" AND docs with "learning"
  - Intersection gives docs containing both terms (anywhere)

With positions:
  - Check if positions differ by exactly 1
  - "machine" at pos 5, "learning" at pos 6 → MATCH
  - "machine" at pos 10, "learning" at pos 50 → NO MATCH
```

**Proximity Scoring**:

```
Query: "covid vaccine"

BM25 + Proximity:
  score = BM25_score × proximity_boost

  proximity_boost = 1 / (1 + avg_distance)

  Example:
    Doc A: "covid" and "vaccine" 2 words apart → boost = 0.33
    Doc B: "covid" and "vaccine" 50 words apart → boost = 0.02
```

#### Implementation Considerations

**Index Size**:

- Current: ~30% of corpus size
- With positions: ~50-60% of corpus size (2x increase)

**Query Speed**:

- Phrase queries: Slightly slower (position checking)
- Proximity scoring: Same speed (positions loaded anyway)

**Memory**:

- Same streaming architecture applies
- Positions stored in posting lists (still on disk)

### 3. Learned Sparse Retrieval

**Motivation**: Bridge gap between traditional IR and neural methods

#### SPLADE Architecture

```
Input: Document text
  ↓
BERT Encoder
  ↓
Token Importance Prediction (per token)
  ↓
Sparse Vector (vocabulary-sized, mostly zeros)
  ↓
Index like traditional terms
```

**Benefit**:

- Neural ranking quality
- Traditional index efficiency
- Interpretable (shows important tokens)

**Integration**:

```cpp
// Offline: Generate sparse representations
for each document:
    sparse_vec = SPLADE(document)
    for token, weight in sparse_vec:
        if weight > threshold:
            add_to_index(token, doc_id, weight)

// Online: Query expansion via SPLADE
query_vec = SPLADE(query)
expanded_terms = top_k_tokens(query_vec)
// Use expanded terms in Boolean retrieval
```

### 4. Multi-Vector Representations (ColBERT)

**Current Bottleneck**: Reranking is sequential (one query-doc pair at a time)

**ColBERT Solution**:

- Pre-compute document embeddings offline
- At query time: only encode query (fast)
- Score = MaxSim between query and document embeddings

```
Indexing Phase:
  for each document:
      doc_embeddings = ColBERT_doc(document)  // [N_tokens × 128]
      store doc_embeddings in index

Query Phase:
  query_embeddings = ColBERT_query(query)  // [M_tokens × 128]

  for each candidate:
      score = MaxSim(query_embeddings, doc_embeddings)
      // For each query token, find max similarity to doc tokens

  sort by score
```

**Speedup**: ~10-100x faster than cross-encoder reranking

**Trade-off**: Slightly lower quality than cross-encoder (~2-3% MAP drop)

### 5. Approximate Nearest Neighbor (ANN) Search

**Use Case**: Fast retrieval for dense vector representations (SPLADE, ColBERT)

#### HNSW (Hierarchical Navigable Small World)

```
Index Structure:
  Layer 2:  o────o────o  (few nodes, long edges)
            │    │    │
  Layer 1:  o─o──o─o──o─o  (more nodes, medium edges)
            │││  │││  │││
  Layer 0:  o─o─o─o─o─o─o─o  (all nodes, short edges)

Search:
  1. Start at top layer
  2. Greedy traverse to nearest neighbor
  3. Drop to next layer
  4. Repeat until bottom layer
  5. Return K nearest neighbors

Time Complexity: O(log N) instead of O(N) for exact search
```

**Integration**:

```cpp
class ANNIndex {
    HNSWIndex hnsw_;

    void add_document(unsigned int doc_id, std::vector<float> embedding) {
        hnsw_.add(doc_id, embedding);
    }

    std::vector<unsigned int> search(std::vector<float> query_embedding, size_t k) {
        return hnsw_.search(query_embedding, k);
    }
};
```

### 6. Hybrid Retrieval

**Combine multiple signals**:

```
Final Score = α × BM25_score + β × Dense_score + γ × Reranker_score

Where:
  BM25_score: Traditional lexical matching
  Dense_score: SPLADE or ColBERT similarity
  Reranker_score: Cross-encoder score
  α, β, γ: Learned weights (optimize on validation set)
```

**Implementation**:

```cpp
class HybridRetriever {
    BooleanRetriever lexical_;
    DenseRetriever dense_;
    NeuralReranker reranker_;

    std::vector<SearchResult> search(std::string query, size_t k) {
        // Stage 1: Lexical retrieval (fast, high recall)
        auto lexical_results = lexical_.search(query, 10000);

        // Stage 2: Dense retrieval (moderate speed, good precision)
        auto dense_results = dense_.search(query, 10000);

        // Stage 3: Merge with learned weights
        auto merged = merge_with_rrf(lexical_results, dense_results);

        // Stage 4: Rerank top-K with cross-encoder (slow, best quality)
        auto reranked = reranker_.rerank(query, merged, k);

        return reranked;
    }
};
```

**Reciprocal Rank Fusion (RRF)**:

```
RRF_score(doc) = Σ (1 / (k + rank_i(doc)))

Where:
  rank_i(doc): Rank of doc in retrieval method i
  k: Constant (typically 60)

Example:
  Doc A: Rank 5 in BM25, Rank 10 in Dense
         → RRF = 1/(60+5) + 1/(60+10) = 0.0154 + 0.0143 = 0.0297

  Doc B: Rank 1 in BM25, Rank 50 in Dense
         → RRF = 1/(60+1) + 1/(60+50) = 0.0164 + 0.0091 = 0.0255

  Doc A ranked higher (better aggregate performance)
```

### 7. Query Understanding

**Current**: Literal term matching with synonym expansion

**Enhancement**: Deep query understanding

#### Query Classification

```cpp
enum QueryType {
    FACTOID,      // "what is covid-19?"
    DEFINITIONAL, // "define machine learning"
    LIST,         // "list symptoms of flu"
    COMPARISON,   // "difference between virus and bacteria"
    PROCEDURAL    // "how to prevent infection"
};

// Route to specialized rankers per type
class QueryRouter {
    QueryType classify(std::string query);
    Ranker* get_ranker(QueryType type);
};
```

#### Query Expansion via LLM

```python
# Generate query variations using GPT/BERT
def expand_query(query: str) -> List[str]:
    variations = llm.generate(
        f"Rephrase this query in 5 different ways: {query}"
    )
    return variations

# Example:
# Input: "covid treatment"
# Output:
#   - "how to treat coronavirus"
#   - "therapies for covid-19"
#   - "medication for sars-cov-2"
#   - "covid-19 patient care"
#   - "coronavirus treatment options"

# Retrieve using all variations, merge results
```

### 8. Incremental Indexing

**Current**: Full rebuild required for new documents

**Enhancement**: Add documents without reindexing

#### Strategy: Delta Index

```
Main Index (large, static)
     +
Delta Index (small, frequently updated)
     =
Merged Results at Query Time
```

**Implementation**:

```cpp
class IncrementalIndexer {
    MainIndex main_index_;      // Immutable, on disk
    DeltaIndex delta_index_;    // In-memory, mutable

    void add_documents(std::vector<Document> new_docs) {
        delta_index_.add(new_docs);

        // When delta grows too large, merge into main
        if (delta_index_.size() > THRESHOLD) {
            merge_delta_into_main();
        }
    }

    ResultSet search(Query q) {
        auto main_results = main_index_.search(q);
        auto delta_results = delta_index_.search(q);
        return merge(main_results, delta_results);
    }
};
```

**Merge Strategy**:

- Background thread merges delta into main
- Atomic swap of main index when complete
- No downtime for queries

### 9. Distributed Indexing and Retrieval

**Motivation**: Scale beyond single machine

#### Architecture

```
Load Balancer
      │
      ├───────────┬───────────┬───────────┐
      ▼           ▼           ▼           ▼
   Node 1      Node 2      Node 3      Node 4
   (docs 0-25K)(docs 25-50K)(docs 50-75K)(docs 75-100K)
      │           │           │           │
      └───────────┴───────────┴───────────┘
                  │
           Merge & Rank Results
```

**Document Partitioning**:

- Hash-based: `node_id = hash(doc_id) % num_nodes`
- Range-based: Node 1 = docs 0-25K, Node 2 = docs 25-50K, ...
- Replication: Each document on 2-3 nodes (fault tolerance)

**Query Processing**:

```
1. Coordinator receives query
2. Broadcast query to all nodes (parallel)
3. Each node returns top-K results
4. Coordinator merges K × num_nodes results
5. Rerank merged results
6. Return final top-K
```

**Challenges**:

- Network latency
- Load balancing (hot shards)
- Consistency during updates
- Fault tolerance

### 10. Caching Layer

**Query Cache**:

```cpp
class QueryCache {
    LRUCache<std::string, std::vector<SearchResult>> cache_;

    std::vector<SearchResult> search(std::string query) {
        if (cache_.contains(query)) {
            return cache_.get(query);  // Cache hit: instant
        }

        auto results = expensive_search(query);
        cache_.put(query, results);
        return results;
    }
};
```

**Posting List Cache**:

```cpp
class PostingCache {
    LRUCache<std::string, PostingList> cache_;
    size_t max_size_ = 10000;  // Cache top 10K terms

    PostingList get_postings(std::string term) {
        if (cache_.contains(term)) {
            return cache_.get(term);
        }

        auto postings = load_from_disk(term);
        cache_.put(term, postings);
        return postings;
    }
};
```

**Document Cache**:

```cpp
// Cache frequently accessed documents in memory
class DocumentCache {
    LRUCache<unsigned int, Document> cache_;

    // Benefits reranking (frequently rerank same docs for different queries)
};
```

---
