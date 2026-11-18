"""
synthesizer.py
Synthesizer agent: groups retrieved papers into themes (KMeans
clustering on embeddings), summarizes each theme (TextRank), and
detects consensus/contradiction/gap signals - all locally, no LLM.
Cache-first: checks for a saved synthesis before recomputing.

Accepts an optional `settings` dict (see config.DEFAULT_SETTINGS),
passed straight through to retriever.retrieve_papers so the papers
synthesized here always match whatever settings the researcher
chose. The cache filename and vector store name reuse the exact
same settings-aware naming as retriever.py, so the Synthesizer
never accidentally loads a paper set fetched under different
settings. The internal retrieve_papers call always uses
force_refresh=False: by the time this stage runs, the Retriever's
own cache has already been written (freshly, if a live re-run was
requested at the top level), so re-forcing a live OpenAlex fetch
here would only be a redundant, slower repeat of the same call.
"""

import os
import json

import config
from retriever import retrieve_papers, _store_name
from vector_store import VectorStore
from textrank import textrank_summarize, split_into_sentences

CONTRADICTION_PAIRS = [
    ("improves", "does not improve"), ("increases", "decreases"),
    ("effective", "ineffective"), ("significant", "not significant"),
    ("outperforms", "underperforms"), ("positive", "negative"),
]
GAP_PHRASES = [
    "future work", "further research", "limited", "remains unclear",
    "not well understood", "few studies", "lack of", "understudied",
]


def _cache_path(topic, settings):
    return os.path.join(config.SYNTHESIZER_CACHE_DIR, f"{_store_name(topic, settings)}.json")


def _detect_gaps(papers):
    """
    Finds sentences that mention gap-signal phrases (e.g. "future
    work", "limited", "lack of"). Returns the FULL sentence each
    phrase appears in, not a fixed-width character slice - avoids
    cutting off mid-word.
    """
    gap_sentences = []
    for p in papers:
        sentences = split_into_sentences(p["abstract"])
        for phrase in GAP_PHRASES:
            found_sentence = None
            for sentence in sentences:
                if phrase in sentence.lower():
                    found_sentence = sentence.strip()
                    break
            if found_sentence:
                gap_sentences.append(found_sentence)
                break  # one gap sentence per paper is enough
    return gap_sentences[:5]


def _detect_contradictions(theme_summaries):
    """Simple keyword-pair heuristic across theme summary sentences."""
    contradictions = []
    all_text = {i: " ".join(s).lower() for i, s in theme_summaries.items()}
    theme_ids = list(all_text.keys())
    for i in range(len(theme_ids)):
        for j in range(i + 1, len(theme_ids)):
            t1, t2 = all_text[theme_ids[i]], all_text[theme_ids[j]]
            for word_a, word_b in CONTRADICTION_PAIRS:
                if word_a in t1 and word_b in t2:
                    contradictions.append(
                        f"Theme {theme_ids[i]} suggests '{word_a}' while Theme {theme_ids[j]} suggests '{word_b}'"
                    )
                elif word_a in t2 and word_b in t1:
                    contradictions.append(
                        f"Theme {theme_ids[j]} suggests '{word_a}' while Theme {theme_ids[i]} suggests '{word_b}'"
                    )
    return contradictions[:5]


def synthesize(topic, settings=None, force_refresh=False):
    merged = config.merge_settings(settings)
    cache_file = _cache_path(topic, merged)

    if os.path.exists(cache_file) and not force_refresh:
        print(f"[synthesizer] Cache hit for '{topic}'")
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)

    # Deliberately not forcing a re-fetch here - the Retriever's own
    # cache is already correct by the time this stage runs (see the
    # module docstring above for why).
    papers = retrieve_papers(topic, settings=merged, force_refresh=False)
    if len(papers) < 2:
        raise ValueError(
            f"Not enough papers with abstracts to synthesize ({len(papers)} found). "
            "Try a broader topic, a lower minimum relevance score, or a wider year range."
        )

    store = VectorStore(_store_name(topic, merged)).load()
    num_clusters = min(config.NUM_THEMES, len(papers))
    labels = store.cluster(num_clusters)

    # Group papers by cluster label
    clusters = {}
    for label, paper in zip(labels, papers):
        clusters.setdefault(label, []).append(paper)

    themes = {}
    theme_summaries = {}
    for label, cluster_papers in clusters.items():
        abstracts = [p["abstract"] for p in cluster_papers]
        summary_sentences = textrank_summarize(abstracts, config.SENTENCES_PER_THEME)
        theme_summaries[label] = summary_sentences
        themes[str(label)] = {
            "paper_titles": [p["title"] for p in cluster_papers],
            "paper_count": len(cluster_papers),
            "summary_sentences": summary_sentences,
        }

    synthesis = {
        "topic": topic,
        "num_papers": len(papers),
        "themes": themes,
        "contradictions": _detect_contradictions(theme_summaries),
        "gaps": _detect_gaps(papers),
    }

    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(synthesis, f, indent=2, ensure_ascii=False)
    print(f"[synthesizer] Cached synthesis to {cache_file}")

    return synthesis


if __name__ == "__main__":
    result = synthesize("machine learning for crop yield prediction")
    print(json.dumps(result, indent=2, ensure_ascii=False))