

import numpy as np
from src.rag.embeddings import model
from src.rag.hybrid_search import HybridSearcher


def retrieve(query, chunks, index, k=3, return_indices=False):
    """
    Standard semantic search
    """
    q_emb = model.encode([query])
    distances, idx = index.search(q_emb, k)

    # Remove duplicates while preserving order
    seen = set()
    unique_indices = []
    for i in idx[0]:
        if i not in seen:
            unique_indices.append(i)
            seen.add(i)

    results = [chunks[i] for i in unique_indices]
    return (results, unique_indices) if return_indices else results


def retrieve_hybrid(query, chunks, index, k=3, alpha=0.5, return_indices=False):
    """
    Hybrid search combining keyword (TF-IDF) and semantic search
    alpha: weight for keyword search (0.5 = 50% keyword, 50% semantic)
    """
    # Semantic search
    q_emb = model.encode([query])
    distances, semantic_idx = index.search(q_emb, k * 2)
    semantic_indices = semantic_idx[0]
    semantic_scores = distances[0]  # L2 distances

    # Hybrid search
    searcher = HybridSearcher(chunks)
    combined_indices = searcher.hybrid_search(
        query, semantic_scores, semantic_indices, k=k, alpha=alpha
    )

    results = [chunks[i] for i in combined_indices]
    return (results, combined_indices) if return_indices else results


def rerank_results(query, results, model_name="all-MiniLM-L6-v2", return_indices=False):
    """
    Re-rank results by relevance using semantic similarity
    """
    from sentence_transformers import util

    query_embedding = model.encode(query)
    result_embeddings = model.encode(results)

    # Calculate similarity scores
    scores = util.pytorch_cos_sim(query_embedding, result_embeddings)[0]

    # Sort by score in descending order
    sorted_pairs = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    ranked_results = [results[i] for i, _ in sorted_pairs]
    ranked_indices = [i for i, _ in sorted_pairs]

    return (ranked_results, ranked_indices) if return_indices else ranked_results
