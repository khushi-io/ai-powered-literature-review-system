"""
retriever.py
Retriever agent: fetches real academic papers from OpenAlex for a
given research topic. Cache-first design (checks a local JSON cache
before calling the live API), with retry-with-backoff for OpenAlex
HTTP 429 (rate limit) errors. Once fetched, papers are deduplicated,
filtered for relevance to the topic (embedding similarity), and
stored in a local VectorStore (vector_store.py) so later stages
(Synthesizer) can search/cluster them.

Every function here accepts an optional `settings` dict (see
config.DEFAULT_SETTINGS / config.merge_settings) controlling how many
papers to fetch and keep, the minimum relevance score, publication
year range, and open-access filtering. Cache filenames and the
vector store name both include a settings-derived suffix, so running
the same topic under different settings never reads or overwrites
the wrong cached result.
"""

import os
import json
import time
import datetime
import requests
import numpy as np

import config
from vector_store import VectorStore, embed_text


def _slugify(text):
    """Turn a topic string into a safe filename, e.g.
    'ML for crop yield!' -> 'ml_for_crop_yield'"""
    safe = "".join(c if c.isalnum() else "_" for c in text.lower())
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe.strip("_")


def _store_name(topic, settings):
    """The identifier used for both the retriever cache file and the
    vector store, so a given (topic, settings) pair always maps to
    the same cached data on both sides."""
    return _slugify(topic) + config.settings_cache_suffix(settings)


def _cache_path(topic, settings):
    return os.path.join(config.RETRIEVER_CACHE_DIR, f"{_store_name(topic, settings)}.json")


def _reconstruct_abstract(inverted_index):
    """
    OpenAlex stores abstracts as an 'inverted index': a dict mapping
    each word to the list of positions it appears at, instead of
    plain text (this saves them space). We rebuild normal text from it.
    Example: {"The": [0], "cat": [1], "sat": [2]} -> "The cat sat"
    """
    if not inverted_index:
        return ""
    position_to_word = {}
    for word, positions in inverted_index.items():
        for pos in positions:
            position_to_word[pos] = word
    ordered_words = [position_to_word[i] for i in sorted(position_to_word.keys())]
    return " ".join(ordered_words)


def _build_filter_string(settings):
    """
    Translates the researcher-facing settings (year range, "only
    recent papers", open-access-only) into an OpenAlex `filter`
    query string. Returns None if no filters apply.
    """
    parts = []

    if settings["recent_only"]:
        this_year = datetime.date.today().year
        from_year = this_year - 3
    else:
        from_year = settings["year_from"]

    to_year = settings["year_to"]

    if from_year:
        parts.append(f"from_publication_date:{int(from_year)}-01-01")
    if to_year:
        parts.append(f"to_publication_date:{int(to_year)}-12-31")
    if settings["open_access_only"]:
        parts.append("is_oa:true")

    return ",".join(parts) if parts else None


def _fetch_from_openalex(topic, settings):
    """
    Calls the OpenAlex API with retry-with-backoff for HTTP 429s.
    Returns a list of cleaned paper dicts. `fetch_limit` (from
    settings) controls how many raw results are pulled per page,
    before deduplication/relevance filtering trims the list down.
    """
    params = {
        "search": topic,
        "per-page": min(settings["fetch_limit"], 200),
        "mailto": config.OPENALEX_MAILTO,
        "api_key": config.OPENALEX_API_KEY,
    }

    filter_string = _build_filter_string(settings)
    if filter_string:
        params["filter"] = filter_string

    last_error = None
    for attempt in range(config.OPENALEX_MAX_RETRIES):
        response = requests.get(config.OPENALEX_API_BASE, params=params, timeout=30)

        if response.status_code == 200:
            data = response.json()
            papers = []
            for work in data.get("results", []):
                abstract = _reconstruct_abstract(work.get("abstract_inverted_index"))
                if not abstract:
                    # Skip papers with no abstract - Synthesizer needs text to work with
                    continue
                papers.append({
                    "id": work.get("id"),
                    "title": work.get("title") or "Untitled",
                    "abstract": abstract,
                    "year": work.get("publication_year"),
                    "authors": [
                        a["author"]["display_name"]
                        for a in work.get("authorships", [])
                        if a.get("author")
                    ],
                    "cited_by_count": work.get("cited_by_count", 0),
                    "doi": work.get("doi"),
                    "is_oa": (work.get("open_access") or {}).get("is_oa", False),
                    "landing_page_url": (work.get("primary_location") or {}).get("landing_page_url"),
                })
            return papers

        elif response.status_code == 429:
            wait_time = config.OPENALEX_BACKOFF_BASE_SECONDS * (2 ** attempt)
            print(f"[retriever] OpenAlex rate limited (429). Waiting {wait_time}s before retry "
                  f"({attempt + 1}/{config.OPENALEX_MAX_RETRIES})...")
            time.sleep(wait_time)
            last_error = "429 Too Many Requests"

        else:
            # Other errors (e.g. 500, 400) - don't retry, fail fast with info
            raise RuntimeError(
                f"OpenAlex request failed with status {response.status_code}: {response.text[:300]}"
            )

    raise RuntimeError(f"OpenAlex still rate-limited after {config.OPENALEX_MAX_RETRIES} retries: {last_error}")


def _deduplicate_papers(papers):
    """Remove papers with near-identical titles (case/whitespace insensitive)."""
    seen_titles = set()
    unique_papers = []
    for p in papers:
        normalized = " ".join(p["title"].lower().split())
        if normalized not in seen_titles:
            seen_titles.add(normalized)
            unique_papers.append(p)
    return unique_papers


def _filter_by_relevance(papers, topic, min_similarity):
    """
    Drop papers whose title+abstract embedding is too dissimilar from
    the topic itself - catches loosely-keyword-matched noise (e.g. an
    unrelated paper OpenAlex returned just because it shared a word).
    A min_similarity of 0.0 skips this step entirely (useful for
    narrow/complex topics where even loosely related papers are
    better than none).
    """
    if min_similarity <= 0.0:
        return papers

    topic_vec = embed_text(topic)
    topic_vec = topic_vec / np.linalg.norm(topic_vec)

    relevant = []
    for p in papers:
        text = f"{p['title']}. {p['abstract']}"
        vec = embed_text(text)
        vec = vec / np.linalg.norm(vec)
        similarity = float(vec @ topic_vec)
        if similarity >= min_similarity:
            relevant.append(p)
        else:
            print(f"[retriever] Filtered out (low relevance {similarity:.2f}): {p['title']}")
    return relevant


def retrieve_papers(topic, settings=None, max_results=None, force_refresh=False):
    """
    Main entry point. Returns a list of paper dicts for the given
    topic, respecting the given settings (or DEFAULT_SETTINGS if
    settings is None). `max_results` is a legacy override kept for
    backward compatibility with older calling code - prefer passing
    settings={"max_papers": N} instead.

    Cache-first: if a cached result exists for this exact
    (topic, settings) pair and force_refresh is False, it's loaded
    instantly with no API call.
    """
    merged = config.merge_settings(settings)
    if max_results is not None:
        merged["max_papers"] = max_results

    cache_file = _cache_path(topic, merged)

    if os.path.exists(cache_file) and not force_refresh:
        print(f"[retriever] Cache hit for topic '{topic}' -> loading from {cache_file}")
        with open(cache_file, "r", encoding="utf-8") as f:
            papers = json.load(f)
        return papers

    print(f"[retriever] No cache found for '{topic}' with these settings. Calling OpenAlex API...")
    papers = _fetch_from_openalex(topic, merged)
    print(f"[retriever] Retrieved {len(papers)} papers with abstracts.")

    papers = _deduplicate_papers(papers)
    papers = _filter_by_relevance(papers, topic, merged["min_relevance"])
    papers = papers[: merged["max_papers"]]
    print(f"[retriever] {len(papers)} papers remaining after dedup + relevance filtering + limit.")

    # Save to cache (JSON, human-readable, easy to inspect/debug)
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)
    print(f"[retriever] Cached results to {cache_file}")

    # Also store in the vector store for later search/clustering
    if papers:
        store = VectorStore(_store_name(topic, merged))
        ids = [p["id"] for p in papers]
        texts = [f"{p['title']}. {p['abstract']}" for p in papers]
        store.add_items(ids, papers, texts)
        store.save()

    return papers


if __name__ == "__main__":
    # Quick manual test when running this file directly:
    # python retriever.py
    test_topic = "machine learning for crop yield prediction"
    results = retrieve_papers(test_topic, force_refresh=True)
    print(f"\nGot {len(results)} papers. Titles:")
    for p in results:
        print("-", p["title"])