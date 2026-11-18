"""
config.py
Central configuration for the Agentic Literature Review project.
No external LLM API is used - all synthesis, fact-checking,
hypothesis generation, and evaluation run on local algorithms
(TF-IDF, TextRank, KMeans clustering, and a local NLI model via
sentence-transformers). The only external API is OpenAlex, used
purely for data retrieval.
"""

import os
import json
import hashlib
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# API KEYS
# ---------------------------------------------------------------------------
OPENALEX_API_KEY = os.getenv("OPENALEX_API_KEY")

if not OPENALEX_API_KEY:
    raise ValueError(
        "OPENALEX_API_KEY not found. Make sure you created a .env file "
        "in the project root with OPENALEX_API_KEY=your_key_here"
    )

# ---------------------------------------------------------------------------
# OPENALEX SETTINGS
# ---------------------------------------------------------------------------
OPENALEX_API_BASE = "https://api.openalex.org/works"
OPENALEX_MAX_RETRIES = 5
OPENALEX_BACKOFF_BASE_SECONDS = 2

# OpenAlex gives faster/more reliable rate limits to requests that
# identify a contact email (the "polite pool").
OPENALEX_MAILTO = "khushichile04@gmail.com"

# ---------------------------------------------------------------------------
# LOCAL MODELS (no API key needed - downloaded once, run offline after)
# ---------------------------------------------------------------------------
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"     # used by vector_store.py
EMBEDDING_DIM = 384
NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-base"  # used by critic.py

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "cache")
VECTOR_STORE_DIR = os.path.join(CACHE_DIR, "vector_store")

RETRIEVER_CACHE_DIR = os.path.join(CACHE_DIR, "retriever")
SYNTHESIZER_CACHE_DIR = os.path.join(CACHE_DIR, "synthesizer")
CRITIC_CACHE_DIR = os.path.join(CACHE_DIR, "critic")
HYPOTHESIS_CACHE_DIR = os.path.join(CACHE_DIR, "hypothesis")
EVALUATOR_CACHE_DIR = os.path.join(CACHE_DIR, "evaluator")
NOVELTY_CACHE_DIR = os.path.join(CACHE_DIR, "novelty")
EXPORTS_DIR = os.path.join(BASE_DIR, "exports")
HISTORY_DIR = os.path.join(BASE_DIR, "history")   # one JSON file per past search (see history.py)

for _dir in [
    CACHE_DIR,
    VECTOR_STORE_DIR,
    RETRIEVER_CACHE_DIR,
    SYNTHESIZER_CACHE_DIR,
    CRITIC_CACHE_DIR,
    HYPOTHESIS_CACHE_DIR,
    EVALUATOR_CACHE_DIR,
    NOVELTY_CACHE_DIR,
    EXPORTS_DIR,
    HISTORY_DIR,
]:
    os.makedirs(_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# PIPELINE SETTINGS (fallback / legacy constants - still used as the
# source of truth for DEFAULT_SETTINGS below)
# ---------------------------------------------------------------------------
MAX_PAPERS_PER_QUERY = 15           # how many papers Retriever keeps after filtering
MAX_CRITIC_REFINEMENT_LOOPS = 3     # core novelty: self-refinement loop cap

NUM_THEMES = 4                      # how many theme clusters to group papers into
SENTENCES_PER_THEME = 3             # how many summary sentences per theme (TextRank)

NLI_CONTRADICTION_THRESHOLD = 0.5   # confidence above which a claim is flagged
CRITIC_APPROVAL_THRESHOLD = 0.7     # support ratio needed for the Critic to approve

NUM_HYPOTHESES = 3                  # how many hypotheses to draft per run

# ---------------------------------------------------------------------------
# NOVELTY CHECKER SETTINGS
# ---------------------------------------------------------------------------
# For each generated hypothesis, the Novelty Checker searches OpenAlex
# using the hypothesis's own grounding terms (not the full topic), so
# it looks specifically for existing work close to THAT idea, rather
# than just re-finding the original topic's paper set.
NOVELTY_SEARCH_RESULTS = 8          # how many candidate papers to compare against, per hypothesis
NOVELTY_SIMILARITY_THRESHOLD = 0.55 # embedding similarity above which a match counts as "close"

# ---------------------------------------------------------------------------
# RESEARCHER-CONTROLLABLE SETTINGS (the Settings Panel)
# ---------------------------------------------------------------------------
# This is the single source of truth for every value a researcher can
# adjust from the desktop app's Settings Panel. Any agent that needs
# one of these values should read it from a merged settings dict
# (see merge_settings below), not from the raw constants above, so
# a researcher's choices are actually respected end to end.
DEFAULT_SETTINGS = {
    "max_papers": MAX_PAPERS_PER_QUERY,   # papers KEPT after dedup/relevance filtering
    "fetch_limit": 25,                    # raw papers PULLED from OpenAlex before filtering
    "min_relevance": 0.25,                # 0.0 = no relevance filtering at all
    "year_from": None,                    # e.g. 2018, or None for no lower bound
    "year_to": None,                      # e.g. 2026, or None for no upper bound
    "recent_only": False,                 # shortcut: overrides year_from to (this year - 3)
    "open_access_only": False,
    "num_hypotheses": NUM_HYPOTHESES,
}

# Sensible bounds used to clamp whatever a researcher enters in the UI,
# so a bad input (e.g. max_papers = 5000) can't cause a runaway request.
SETTINGS_BOUNDS = {
    "max_papers": (3, 30),
    "fetch_limit": (5, 50),
    "min_relevance": (0.0, 0.6),
    "num_hypotheses": (1, 6),
}


def _clamp(value, low, high):
    try:
        value = type(low)(value)
    except (TypeError, ValueError):
        return low
    return max(low, min(high, value))


def merge_settings(user_settings=None):
    """
    Combines whatever a researcher/UI provided with DEFAULT_SETTINGS,
    filling in anything missing and clamping numeric fields to sane
    bounds. Always returns a complete, safe settings dict - every
    agent function should call this on whatever `settings` argument
    it receives (including None) before using it.
    """
    merged = dict(DEFAULT_SETTINGS)
    if user_settings:
        for key in DEFAULT_SETTINGS:
            if key in user_settings and user_settings[key] is not None:
                merged[key] = user_settings[key]

    for key, (low, high) in SETTINGS_BOUNDS.items():
        merged[key] = _clamp(merged[key], low, high)

    # fetch_limit should never be smaller than max_papers, or the
    # Retriever would be asked to keep more papers than it fetched.
    if merged["fetch_limit"] < merged["max_papers"]:
        merged["fetch_limit"] = merged["max_papers"]

    return merged


def settings_cache_suffix(settings=None):
    """
    Returns a short, stable filename suffix derived from a settings
    dict, so that running the SAME topic under DIFFERENT settings
    never reads or overwrites the wrong cache file. Returns an empty
    string when settings match the defaults exactly, so all of the
    project's existing cached demo topics (cached before the Settings
    Panel existed) keep working without needing to be re-fetched.
    """
    merged = merge_settings(settings)
    if merged == DEFAULT_SETTINGS:
        return ""
    key = json.dumps(merged, sort_keys=True)
    return "_" + hashlib.md5(key.encode("utf-8")).hexdigest()[:8]