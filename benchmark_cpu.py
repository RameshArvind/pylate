#!/usr/bin/env python
"""CPU benchmarking script for pylate library.

Benchmarks encoding, indexing, and retrieval operations on CPU.
"""

from __future__ import annotations

import time
import numpy as np
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


def benchmark_encoding(
    model: Any,
    num_documents: int = 1000,
    num_queries: int = 100,
    batch_size: int = 64,
) -> dict[str, Any]:
    """Benchmark encoding performance.

    Parameters
    ----------
    model : Any
        The ColBERT model instance.
    num_documents : int
        Number of documents to encode.
    num_queries : int
        Number of queries to encode.
    batch_size : int
        Batch size for encoding.

    Returns
    -------
    dict[str, Any]
        Dictionary containing encoding benchmark results.
    """
    print("\n" + "=" * 60)
    print("ENCODING BENCHMARK")
    print("=" * 60)

    # Generate synthetic documents and queries
    documents = [
        f"This is document number {i}. It contains information about various topics in machine learning and information retrieval."
        for i in range(num_documents)
    ]
    queries = [
        f"Query {i}: What is information about topic {i % 10}?"
        for i in range(num_queries)
    ]

    print(f"\nDocuments to encode: {num_documents}")
    print(f"Queries to encode: {num_queries}")
    print(f"Batch size: {batch_size}")

    # Benchmark document encoding
    print(f"\nEncoding {num_documents} documents...")
    start = time.time()
    doc_embeddings = model.encode(
        sentences=documents,
        batch_size=batch_size,
        is_query=False,
        show_progress_bar=False,
    )
    doc_time = time.time() - start
    doc_throughput = num_documents / doc_time

    print(f"  Time: {format_time(doc_time)}")
    print(f"  Throughput: {doc_throughput:.2f} docs/sec")
    if hasattr(doc_embeddings, 'shape'):
        print(f"  Embedding shape: {doc_embeddings.shape}")
    elif isinstance(doc_embeddings, list):
        print(f"  Embeddings count: {len(doc_embeddings)}")

    # Benchmark query encoding
    print(f"\nEncoding {num_queries} queries...")
    start = time.time()
    query_embeddings = model.encode(
        sentences=queries,
        batch_size=batch_size,
        is_query=True,
        show_progress_bar=False,
    )
    query_time = time.time() - start
    query_throughput = num_queries / query_time

    print(f"  Time: {format_time(query_time)}")
    print(f"  Throughput: {query_throughput:.2f} queries/sec")
    if hasattr(query_embeddings, 'shape'):
        print(f"  Embedding shape: {query_embeddings.shape}")
    elif isinstance(query_embeddings, list):
        print(f"  Embeddings count: {len(query_embeddings)}")

    return {
        "document_encoding_time": doc_time,
        "document_throughput": doc_throughput,
        "query_encoding_time": query_time,
        "query_throughput": query_throughput,
        "total_time": doc_time + query_time,
    }


def benchmark_indexing_and_retrieval(
    model: Any,
    num_documents: int = 1000,
    num_queries: int = 100,
    k: int = 10,
    batch_size: int = 64,
) -> dict[str, Any]:
    """Benchmark indexing and retrieval performance.

    Parameters
    ----------
    model : Any
        The ColBERT model instance.
    num_documents : int
        Number of documents to index.
    num_queries : int
        Number of queries to retrieve for.
    k : int
        Number of top results to retrieve.
    batch_size : int
        Batch size for encoding.

    Returns
    -------
    dict[str, Any]
        Dictionary containing indexing and retrieval benchmark results.
    """
    print("\n" + "=" * 60)
    print("INDEXING AND RETRIEVAL BENCHMARK")
    print("=" * 60)

    # Generate synthetic data
    documents = [
        f"Document {i}: Information about machine learning, neural networks, and deep learning techniques."
        for i in range(num_documents)
    ]
    queries = [
        f"Query {i}: How does deep learning work?"
        for i in range(num_queries)
    ]

    print(f"\nDocuments: {num_documents}")
    print(f"Queries: {num_queries}")
    print(f"Top-k: {k}")
    print(f"Batch size: {batch_size}")

    # Encode documents
    print(f"\nEncoding {num_documents} documents...")
    start = time.time()
    doc_embeddings = model.encode(
        sentences=documents,
        batch_size=batch_size,
        is_query=False,
        show_progress_bar=False,
    )
    encoding_time = time.time() - start
    print(f"  Time: {format_time(encoding_time)}")

    # Create index
    print(f"\nCreating PLAID index...")
    start = time.time()
    index = indexes.PLAID(override=True, index_name="benchmark_index")
    index_creation_time = time.time() - start
    print(f"  Time: {format_time(index_creation_time)}")

    # Add documents to index
    print(f"\nAdding {num_documents} documents to index...")
    doc_ids = [f"doc_{i}" for i in range(num_documents)]
    start = time.time()
    index.add_documents(
        documents_ids=doc_ids,
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
        batch_size=batch_size,
        is_query=True,
        show_progress_bar=False,
    )
    query_encoding_time = time.time() - start
    print(f"  Time: {format_time(query_encoding_time)}")

    # Create retriever and retrieve
    print(f"\nCreating retriever...")
    retriever = retrieve.ColBERT(index=index)

    print(f"Retrieving top-{k} results for {num_queries} queries...")
    start = time.time()
    scores = retriever.retrieve(
        queries_embeddings=query_embeddings,
        k=k,
    )
    retrieval_time = time.time() - start
    retrieval_throughput = num_queries / retrieval_time

    print(f"  Time: {format_time(retrieval_time)}")
    print(f"  Throughput: {retrieval_throughput:.2f} queries/sec")
    print(f"  Results sample: {scores[0][:3] if scores else 'No results'}")

    return {
        "encoding_time": encoding_time,
        "index_creation_time": index_creation_time,
        "indexing_time": indexing_time,
        "query_encoding_time": query_encoding_time,
        "retrieval_time": retrieval_time,
        "retrieval_throughput": retrieval_throughput,
        "total_time": (
            encoding_time
            + index_creation_time
            + indexing_time
            + query_encoding_time
            + retrieval_time
        ),
    }


def benchmark_scaling(
    model: Any,
    sizes: list[int] = [100, 500, 1000, 5000],
    num_queries: int = 50,
    k: int = 10,
) -> dict[str, Any]:
    """Benchmark how performance scales with corpus size.

    Parameters
    ----------
    model : Any
        The ColBERT model instance.
    sizes : list[int]
        List of corpus sizes to test.
    num_queries : int
        Number of queries per size.
    k : int
        Number of top results to retrieve.

    Returns
    -------
    dict[str, Any]
        Dictionary containing scaling benchmark results.
    """
    print("\n" + "=" * 60)
    print("SCALING BENCHMARK")
    print("=" * 60)

    scaling_results = {}

    for size in sizes:
        print(f"\n--- Testing with {size} documents ---")

        documents = [
            f"Document {i}: Content about machine learning and information retrieval."
            for i in range(size)
        ]
        queries = [
            f"Query {i}: Information retrieval question?"
            for i in range(num_queries)
        ]

        # Encode and index
        doc_embeddings = model.encode(
            sentences=documents,
            batch_size=64,
            is_query=False,
            show_progress_bar=False,
        )

        index = indexes.PLAID(
            override=True,
            index_name=f"benchmark_scale_{size}",
        )
        doc_ids = [f"doc_{i}" for i in range(size)]

        start = time.time()
        index.add_documents(
            documents_ids=doc_ids,
            documents_embeddings=doc_embeddings,
        )
        indexing_time = time.time() - start

        # Encode queries and retrieve
        query_embeddings = model.encode(
            sentences=queries,
            batch_size=32,
            is_query=True,
            show_progress_bar=False,
        )

        retriever = retrieve.ColBERT(index=index)

        start = time.time()
        scores = retriever.retrieve(
            queries_embeddings=query_embeddings,
            k=k,
        )
        retrieval_time = time.time() - start

        throughput = num_queries / retrieval_time

        print(f"  Indexing time: {format_time(indexing_time)}")
        print(f"  Retrieval time: {format_time(retrieval_time)}")
        print(f"  Throughput: {throughput:.2f} queries/sec")

        scaling_results[size] = {
            "indexing_time": indexing_time,
            "retrieval_time": retrieval_time,
            "throughput": throughput,
        }

    return scaling_results


def print_summary(results: dict[str, Any]) -> None:
    """Print a summary of all benchmark results."""
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)

    for key, value in results.items():
        if isinstance(value, dict):
            print(f"\n{key}:")
            for k, v in value.items():
                if isinstance(v, float):
                    if "throughput" in k or "throughput" in k.lower():
                        print(f"  {k}: {v:.2f} items/sec")
                    else:
                        print(f"  {k}: {format_time(v)}")
                else:
                    print(f"  {k}: {v}")


if __name__ == "__main__":
    print("=" * 60)
    print("PYLATE CPU BENCHMARKING SUITE")
    print("=" * 60)

    # Load model
    print("\nLoading ColBERT model...")
    model_name = "lightonai/GTE-ModernColBERT-v1"
    model = models.ColBERT(
        model_name_or_path=model_name,
        document_length=300,
        query_length=32,
    )
    print(f"Model loaded: {model_name}")

    # Run benchmarks
    all_results = {}

    # 1. Encoding benchmark
    encoding_results = benchmark_encoding(model, num_documents=1000, num_queries=100)
    all_results["encoding"] = encoding_results

    # 2. Indexing and retrieval benchmark
    indexing_results = benchmark_indexing_and_retrieval(
        model, num_documents=5000, num_queries=100, k=10
    )
    all_results["indexing_and_retrieval"] = indexing_results

    # 3. Scaling benchmark
    scaling_results = benchmark_scaling(model, sizes=[100, 500, 1000, 5000])
    all_results["scaling"] = scaling_results

    # Print summary
    print_summary(all_results)

    print("\n" + "=" * 60)
    print("Benchmark completed!")
    print("=" * 60)
