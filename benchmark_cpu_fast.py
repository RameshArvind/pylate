#!/usr/bin/env python
"""Fast CPU benchmarking script using AnswerAI small ColBERT model.

This script uses a 30M parameter model for quick benchmarking.
"""

from __future__ import annotations

import time
from typing import Any

from pylate import models, indexes, retrieve


def format_time(seconds: float) -> str:
    """Format time in seconds to human-readable format."""
    if seconds < 1e-3:
        return f"{seconds * 1e6:.2f} µs"
    elif seconds < 1:
        return f"{seconds * 1e3:.2f} ms"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        if hours > 0:
            return f"{hours}h {minutes}m {secs:.2f}s"
        elif minutes > 0:
            return f"{minutes}m {secs:.2f}s"
        else:
            return f"{secs:.2f}s"


def quick_benchmark(
    model_name: str,
    num_documents: int = 500,
    num_queries: int = 50,
    k: int = 10,
    skip_retrieval: bool = False,
) -> dict[str, Any]:
    """Run a quick benchmark with specified model.

    Parameters
    ----------
    model_name : str
        Model name or path.
    num_documents : int
        Number of documents to index.
    num_queries : int
        Number of queries to retrieve.
    k : int
        Top-k results to retrieve.
    skip_retrieval : bool
        Skip retrieval phase (for incompatible models).

    Returns
    -------
    dict[str, Any]
        Benchmark results.
    """
    print(f"\n{'=' * 60}")
    print(f"QUICK BENCHMARK - {model_name.split('/')[-1]}")
    print(f"{'=' * 60}")

    print(f"\nLoading model: {model_name}")
    start = time.time()
    model = models.ColBERT(
        model_name_or_path=model_name,
        document_length=300,
        query_length=32,
    )
    load_time = time.time() - start
    print(f"  Load time: {format_time(load_time)}")

    # Generate data
    print(f"\nGenerating {num_documents} documents and {num_queries} queries...")
    documents = [
        f"Document {i}: This is a sample document about machine learning and information retrieval systems."
        for i in range(num_documents)
    ]
    queries = [
        f"Query {i}: What is information retrieval?"
        for i in range(num_queries)
    ]

    # Encode documents
    print(f"\nEncoding {num_documents} documents...")
    start = time.time()
    doc_embeddings = model.encode(
        sentences=documents,
        batch_size=64,
        is_query=False,
        show_progress_bar=False,
    )
    doc_encode_time = time.time() - start
    doc_throughput = num_documents / doc_encode_time
    print(f"  Time: {format_time(doc_encode_time)}")
    print(f"  Throughput: {doc_throughput:.2f} docs/sec")

    # Create index and add documents
    print(f"\nCreating PLAID index and adding documents...")
    start = time.time()
    index = indexes.PLAID(override=True, index_name="fast_benchmark")
    index.add_documents(
        documents_ids=[f"doc_{i}" for i in range(num_documents)],
        documents_embeddings=doc_embeddings,
    )
    indexing_time = time.time() - start
    print(f"  Time: {format_time(indexing_time)}")
    print(f"  Throughput: {num_documents / indexing_time:.2f} docs/sec")

    # Encode queries
    print(f"\nEncoding {num_queries} queries...")
    start = time.time()
    query_embeddings = model.encode(
        sentences=queries,
        batch_size=32,
        is_query=True,
        show_progress_bar=False,
    )
    query_encode_time = time.time() - start
    print(f"  Time: {format_time(query_encode_time)}")

    # Retrieve
    retrieval_time = 0
    retrieval_throughput = 0
    if not skip_retrieval:
        print(f"\nRetrieving top-{k} results...")
        retriever = retrieve.ColBERT(index=index)
        start = time.time()
        scores = retriever.retrieve(
            queries_embeddings=query_embeddings,
            k=k,
        )
        retrieval_time = time.time() - start
        retrieval_throughput = num_queries / retrieval_time
        print(f"  Time: {format_time(retrieval_time)}")
        print(f"  Throughput: {retrieval_throughput:.2f} queries/sec")

        # Show sample result
        if scores and scores[0]:
            print(f"  Sample results for first query: {scores[0][:3]}")
    else:
        print(f"\nSkipping retrieval phase (model compatibility issue)")

    total_time = load_time + doc_encode_time + indexing_time + query_encode_time + retrieval_time

    return {
        "model": model_name,
        "load_time": load_time,
        "document_encoding_time": doc_encode_time,
        "document_throughput": doc_throughput,
        "indexing_time": indexing_time,
        "query_encoding_time": query_encode_time,
        "retrieval_time": retrieval_time,
        "retrieval_throughput": retrieval_throughput,
        "total_time": total_time,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("PYLATE FAST CPU BENCHMARKING (Small Models)")
    print("=" * 60)

    results = {}

    # Test with small model (30M parameters)
    try:
        print("\n[1/2] Testing AnswerAI Small ColBERT (30M parameters)...")
        small_results = quick_benchmark(
            model_name="lightonai/answerai-colbert-small-v1",
            num_documents=500,
            num_queries=50,
            skip_retrieval=True,  # Skip retrieval due to model compatibility
        )
        results["small"] = small_results
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {str(e)[:100]}...")
        print("  Skipping small model...")

    # Test with default model for comparison
    try:
        print("\n[2/2] Testing GTE-ModernColBERT (larger model for comparison)...")
        large_results = quick_benchmark(
            model_name="lightonai/GTE-ModernColBERT-v1",
            num_documents=500,
            num_queries=50,
        )
        results["large"] = large_results
    except Exception as e:
        print(f"  ERROR: {e}")

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")

    for model_type, result in results.items():
        print(f"\n{model_type.upper()} MODEL:")
        print(f"  Model: {result['model']}")
        print(f"  Load time: {format_time(result['load_time'])}")
        print(f"  Document encoding: {format_time(result['document_encoding_time'])} ({result['document_throughput']:.2f} docs/sec)")
        print(f"  Indexing: {format_time(result['indexing_time'])}")
        print(f"  Query encoding: {format_time(result['query_encoding_time'])}")
        print(f"  Retrieval: {format_time(result['retrieval_time'])} ({result['retrieval_throughput']:.2f} queries/sec)")
        print(f"  Total: {format_time(result['total_time'])}")

    if len(results) == 2:
        print(f"\n{'=' * 60}")
        print("COMPARISON")
        print(f"{'=' * 60}")
        small_total = results["small"]["total_time"]
        large_total = results["large"]["total_time"]
        speedup = large_total / small_total
        print(f"\nSmall model is {speedup:.1f}x faster than large model")
        print(f"  Small total: {format_time(small_total)}")
        print(f"  Large total: {format_time(large_total)}")

    print("\n" + "=" * 60)
    print("Benchmark completed!")
    print("=" * 60)
