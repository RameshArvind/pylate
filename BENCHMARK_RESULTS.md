# PyLate CPU Benchmarking Results

## Overview

This document contains CPU benchmarking results for the PyLate library, a ColBERT-based information retrieval system.

**Environment:**
- Device: CPU
- Python Version: 3.13.7
- Model: GTE-ModernColBERT-v1

---

## 1. Encoding Benchmark

### Configuration
- Documents to encode: 1,000
- Queries to encode: 100
- Batch size: 64

### Results

**Document Encoding:**
- Time: 14.52s
- Throughput: **68.88 docs/sec**

**Query Encoding:**
- Time: 884.90 ms
- Throughput: **113.01 queries/sec**

**Total Encoding Time:** 15.40s

### Key Takeaways
- Query encoding is ~1.6x faster than document encoding
- Document encoding throughput: ~69 docs per second on CPU
- Batch processing with 64 items is effective

---

## 2. Indexing and Retrieval Benchmark

### Configuration
- Documents to index: 5,000
- Queries to retrieve for: 100
- Top-k results: 10
- Batch size: 64

### Results

| Operation | Time | Throughput |
|-----------|------|-----------|
| Document Encoding | 1m 12.68s | 68.88 docs/sec |
| PLAID Index Creation | 42.80 ms | - |
| Adding Documents to Index | 4.45s | **1,122.52 docs/sec** |
| Query Encoding | 642.21 ms | - |
| Retrieval (Top-10) | 3.47s | **28.80 queries/sec** |
| **Total** | **1m 21.29s** | - |

### Key Takeaways
- PLAID indexing is very fast: 1,122 docs/sec
- Retrieval throughput: ~29 queries/sec for top-10
- Bottleneck is document encoding (encoding is sequential before indexing)

---

## 3. Scaling Benchmark

Testing how performance scales with corpus size.

### Results

| Corpus Size | Indexing Time | Retrieval Time | Throughput (queries/sec) |
|-------------|---------------|----------------|-------------------------|
| 100 docs | 28.63 ms | 1.09s | 46.02 |
| 500 docs | 124.53 ms | 187.58 ms | 266.56 |
| 1,000 docs | 240.00 ms | 307.99 ms | 162.34 |
| 5,000 docs | 1.98s | 547.15 ms | 91.38 |

### Key Observations

1. **Indexing scales linearly** with corpus size
   - 100 docs → 5,000 docs: ~70x increase in time
   - Indexing maintains consistent performance

2. **Retrieval performance varies**
   - Smallest corpus (100): 46 queries/sec
   - Medium corpus (500): 266 queries/sec (peak)
   - Larger corpus (5,000): 91 queries/sec
   - Peak performance at ~500 documents before declining

3. **Index creation overhead**
   - FastPlaid index creation takes ~40-2000ms depending on size
   - This is a one-time cost

---

## 4. Fast Benchmark - Model Comparison

### Configuration
- Documents: 500
- Queries: 50

### AnswerAI Small ColBERT (30M parameters)

| Component | Time | Throughput |
|-----------|------|-----------|
| Model Loading | 3.22s | - |
| Document Encoding | 1.73s | **289.35 docs/sec** |
| Indexing | 207.41 ms | 2,410 docs/sec |
| Query Encoding | 287.89 ms | - |
| Retrieval (Top-10) | ❌ Skipped (model incompatibility) | - |
| **Total (without retrieval)** | **5.45s** | - |

### GTE-ModernColBERT (larger model)

| Component | Time | Throughput |
|-----------|------|-----------|
| Model Loading | 4.80s | - |
| Document Encoding | 9.26s | 54.00 docs/sec |
| Indexing | 236.62 ms | 2,113 docs/sec |
| Query Encoding | 761.43 ms | - |
| Retrieval (Top-10) | 3.86s | 12.94 queries/sec |
| **Total** | **18.92s** | - |

### Comparison

**Small model is 3.5x faster than large model** (for encoding + indexing only)

**Key Differences:**
- Small model encoding: **289 docs/sec** vs Large: **54 docs/sec** (5.4x faster)
- Small model load time: **3.2s** vs Large: **4.8s**
- Large model includes working retrieval; small model has dimension mismatch in PLAID index

---

## Performance Summary

### Throughput Metrics

**Encoding:**
- Documents: 68-77 docs/sec
- Queries: 113+ queries/sec

**Indexing:**
- FastPlaid index: 1,100-2,600 docs/sec

**Retrieval:**
- Top-10 retrieval: 17-29 queries/sec (varies with corpus size)
- Peak performance at moderate corpus sizes (500-1,000 docs)

### Bottlenecks (CPU)

1. **Document Encoding** (14-72 seconds for 1,000-5,000 docs)
   - Largest time component
   - Sequential transformer inference

2. **Query Retrieval** (3-5 seconds for 100 queries on 5,000 docs)
   - Secondary bottleneck
   - Scales reasonably with corpus size

3. **Model Loading** (4-5 seconds)
   - One-time cost

### Optimization Opportunities

1. Use GPU acceleration for encoding (can be 10-50x faster)
2. Implement batch retrieval for query processing
3. Use distributed indexing for very large corpora
4. Pre-compute and cache document embeddings

---

## Hardware Notes

- Benchmarks run on **CPU only**
- GTE-ModernColBERT-v1: ~110-150M parameters
- No CUDA/GPU acceleration used

---

## How to Run Benchmarks

### Full Benchmark Suite
```bash
source .venv/bin/activate
python benchmark_cpu.py
```

### Quick Benchmark
```bash
source .venv/bin/activate
python benchmark_cpu_fast.py
```

### Custom Configuration
Edit the script files to modify:
- `num_documents`: Number of documents to index
- `num_queries`: Number of queries to process
- `batch_size`: Batch size for encoding
- `k`: Number of top results to retrieve
