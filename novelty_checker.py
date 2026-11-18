"""
novelty_checker.py
Novelty Checker agent: for each generated hypothesis, searches
OpenAlex again - but with a narrower, targeted query built from the
hypothesis's own grounding terms, not the original topic - and
compares the hypothesis text against the candidate papers found
using the same local embeddings used throughout this project. No
external LLM API is used.

IMPORTANT HONESTY NOTE: this can only report "no close match was
found in what was searched" - it can never certify that an idea is
genuinely, globally novel. A hypothesis with no close match simply
means nothing similar turned up in the OpenAlex papers this check
queried, not that the idea is proven original.

Reuses the exact same OpenAlex client, caching, and retry-with-
backoff logic already built in retriever.py - the broader search
here is just retrieve_papers() called with the hypothesis's
grounding terms as the "topic" string, so it benefits from the same
cache-first design as every other agent.
"""

import os
import json
import numpy as np

import config
from retriever import retrieve_papers, _store_name
from vector_store import embed_text


def _cache_path(topic, settings):
    return os.path.join(config.NOVELTY_CACHE_DIR, f"{_store_name(topic, settings)}.json")


def _build_query(hypothesis_obj):
    """
    Builds the search query for a single hypothesis from its own
    grounding terms (the specific gap-derived words it was built
    from), not the full topic - this targets the search at the
    hypothesis's actual content rather than just re-finding the
    original topic's paper set.
    """
    terms = hypothesis_obj.get("grounding_terms", [])
    return " ".join(terms) if terms else hypothesis_obj["hypothesis"][:60]


def _check_single_hypothesis(hypothesis_obj, force_refresh=False):
    """
    Searches OpenAlex for candidates matching this hypothesis's
    grounding terms, embeds the hypothesis and each candidate, and
    returns the most similar candidate found (if any), with a
    similarity score and a cautious verdict.
    """
    query = _build_query(hypothesis_obj)

    candidate_settings = {
        "max_papers": config.NOVELTY_SEARCH_RESULTS,
        "fetch_limit": config.NOVELTY_SEARCH_RESULTS,
        "min_relevance": 0.0,  # we want raw candidates here, not pre-filtered by the topic embedding
    }

    try:
        candidates = retrieve_papers(query, settings=candidate_settings, force_refresh=force_refresh)
    except Exception as e:
        return {
            "hypothesis_id": hypothesis_obj["id"],
            "query_used": query,
            "checked": False,
            "reason": f"Search failed: {e}",
        }

    if not candidates:
        return {
            "hypothesis_id": hypothesis_obj["id"],
            "query_used": query,
            "checked": True,
            "best_match_title": None,
            "similarity": 0.0,
            "verdict": "No candidate papers were found for this query.",
        }

    hyp_vec = embed_text(hypothesis_obj["hypothesis"])
    hyp_vec = hyp_vec / np.linalg.norm(hyp_vec)

    best_paper = None
    best_similarity = -1.0
    for paper in candidates:
        text = f"{paper['title']}. {paper['abstract']}"
        vec = embed_text(text)
        vec = vec / np.linalg.norm(vec)
        similarity = float(vec @ hyp_vec)
        if similarity > best_similarity:
            best_similarity = similarity
            best_paper = paper

    is_close_match = best_similarity >= config.NOVELTY_SIMILARITY_THRESHOLD
    verdict = (
        "Closely resembles existing published work - review the match before treating this as a new idea."
        if is_close_match
        else "No closely similar work was found in what was searched; this does not guarantee the idea is novel."
    )

    return {
        "hypothesis_id": hypothesis_obj["id"],
        "query_used": query,
        "checked": True,
        "is_close_match": is_close_match,
        "similarity": round(best_similarity, 3),
        "best_match_title": best_paper["title"],
        "best_match_year": best_paper.get("year"),
        "best_match_authors": best_paper.get("authors", [])[:3],
        "best_match_url": best_paper.get("landing_page_url") or best_paper.get("doi"),
        "verdict": verdict,
    }


def check_novelty(hyp_result, settings=None, force_refresh=False):
    """
    Main entry point. Takes the Hypothesis Generator's result (from
    hypothesis_generator.generate_hypotheses) and checks each
    hypothesis for close resemblance to already-published work.
    Cache-first, keyed by the same (topic, settings) as the other
    stages, so this doesn't re-run every hypothesis's search on
    every repeated pipeline run.
    """
    topic = hyp_result["topic"]
    merged = config.merge_settings(settings)
    cache_file = _cache_path(topic, merged)

    if os.path.exists(cache_file) and not force_refresh:
        print(f"[novelty_checker] Cache hit for '{topic}'")
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)

    per_hypothesis_results = []
    for hypothesis_obj in hyp_result["hypotheses"]:
        print(f"[novelty_checker] Checking {hypothesis_obj['id']} against a broader OpenAlex search...")
        result = _check_single_hypothesis(hypothesis_obj, force_refresh=force_refresh)
        per_hypothesis_results.append(result)

    checked_count = sum(1 for r in per_hypothesis_results if r.get("checked"))
    close_match_count = sum(1 for r in per_hypothesis_results if r.get("is_close_match"))

    summary = {
        "topic": topic,
        "num_hypotheses_checked": checked_count,
        "num_close_matches_found": close_match_count,
        "per_hypothesis": per_hypothesis_results,
    }

    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[novelty_checker] Cached novelty check to {cache_file}")

    return summary


if __name__ == "__main__":
    from synthesizer import synthesize
    from critic import run_critic_loop
    from hypothesis_generator import generate_hypotheses

    topic = "machine learning for crop yield prediction"
    synthesis = synthesize(topic)
    critic_result = run_critic_loop(synthesis, topic)
    hyp_result = generate_hypotheses(critic_result)

    novelty_result = check_novelty(hyp_result, force_refresh=True)

    print("\n=== NOVELTY CHECK RESULTS ===")
    for r in novelty_result["per_hypothesis"]:
        print(f"\n[{r['hypothesis_id']}] query: '{r.get('query_used')}'")
        if r.get("checked"):
            print(f"  Best match: {r.get('best_match_title')} ({r.get('best_match_year')})")
            print(f"  Similarity: {r.get('similarity')}")
            print(f"  Verdict: {r.get('verdict')}")
        else:
            print(f"  Not checked: {r.get('reason')}")