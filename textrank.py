"""
textrank.py
Graph-based extractive summarization (TextRank - same family of
algorithm as Google PageRank, applied to sentences instead of
webpages). Used by synthesizer.py to pick the most representative
sentences from a group of paper abstracts, without calling any LLM.

PageRank works by iteratively refining sentence scores until they
stabilize. For some sentence sets - typically small, narrow, or
repetitive ones (common with niche topics) - this iteration can
oscillate instead of settling, raising
networkx.PowerIterationFailedConvergence. This module first retries
with more iterations and a looser tolerance, and if that still
doesn't converge, falls back to a simple, non-iterative centrality
measure (summed similarity to all other sentences) that always
produces an answer, just a slightly less refined one.
"""

import re
import numpy as np
import networkx as nx
from vector_store import embed_texts


def split_into_sentences(text):
    """Basic sentence splitter (good enough for academic abstracts)."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 20]


def textrank_summarize(texts, num_sentences=3):
    """
    texts: list of abstract strings (one cluster's worth of papers).
    Returns the top `num_sentences` most representative sentences,
    picked via TextRank where possible, falling back to a simpler
    similarity-sum ranking if PageRank fails to converge.
    """
    all_sentences = []
    for text in texts:
        all_sentences.extend(split_into_sentences(text))

    if len(all_sentences) <= num_sentences:
        return all_sentences

    # Embed every sentence, build a similarity graph
    embeddings = embed_texts(all_sentences)
    norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    similarity_matrix = norm @ norm.T

    try:
        graph = nx.from_numpy_array(similarity_matrix)
        # More iterations and a looser tolerance than the default
        # (100 iterations, 1e-6) give PageRank a better chance to
        # converge on small/repetitive sentence sets before we give up.
        scores = nx.pagerank(graph, max_iter=1000, tol=1e-4)
    except nx.PowerIterationFailedConvergence:
        print("[textrank] PageRank did not converge for this sentence set; "
              "falling back to similarity-sum ranking instead.")
        # Fallback: rank each sentence by how similar it is to every
        # other sentence, summed. This is a simpler proxy for
        # "central/representative" that requires no iteration, so it
        # can never fail to converge - it always produces an answer.
        similarity_sums = similarity_matrix.sum(axis=1)
        scores = {i: float(similarity_sums[i]) for i in range(len(all_sentences))}

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_indices = [idx for idx, _ in ranked[:num_sentences]]
    top_indices.sort()  # keep original order for readability

    return [all_sentences[i] for i in top_indices]