"""
comparison.py
Side-by-side comparison of two finished literature-review runs
("search A" and "search B"), taken from the Search History.

Nothing is fetched and no pipeline stage is re-run: both inputs are
reports that were already produced and saved, so a comparison is
instant. Everything here is local and classical - set operations,
the project's own sentence-embedding model (all-MiniLM-L6-v2, the
same one the Synthesizer and Novelty Checker use) and cosine
similarity. No LLM and no external API is involved.

What is compared
----------------
1. Headline statistics   - papers, themes, hypotheses, critic verdict,
                           year range, citations, open-access share.
2. Shared papers         - papers that appear in BOTH searches, matched
                           by DOI (preferred) or by normalized title.
3. Key terms             - which extracted key terms the searches share
                           and which are unique to each.
4. Theme alignment       - every theme of A is compared with every
                           theme of B by the cosine similarity of their
                           embeddings. For each theme of A the closest
                           theme of B is reported and labelled:
                              aligned  (similarity >= 0.55)
                              related  (similarity >= 0.35)
                              distinct (below that)
                           Themes of B that no theme of A resembles are
                           listed as "unique to B".
5. A plain-language summary written from a fixed template (no LLM).

Honest-framing note: the similarity labels are INDICATIVE. The two
thresholds are sensible defaults (the "aligned" value matches the
Novelty Checker's threshold) and can be changed below, but a score is
a measure of textual/semantic closeness, not proof that two themes
are "the same research question".

Robustness: if the embedding model cannot be used for any reason,
the theme comparison automatically falls back to TF-IDF cosine
similarity (scikit-learn) and says so in the result's "method" field
instead of failing.
"""

import re

import numpy as np

import config

# Tunable thresholds for labelling theme-to-theme similarity.
THEME_ALIGNED_THRESHOLD = 0.55
THEME_RELATED_THRESHOLD = 0.35

_LABEL_MAX_CHARS = 90
_MAX_SHARED_PAPERS_LISTED = 50


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _normalize_doi(value):
    """'https://doi.org/10.1/ABC' and '10.1/abc' both become '10.1/abc'."""
    if not value:
        return ""
    text = str(value).strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "https://dx.doi.org/",
                   "http://dx.doi.org/", "doi:"):
        if text.startswith(prefix):
            text = text[len(prefix):]
    return text.strip()


def _normalize_title(value):
    """Lower-case letters and digits only, so punctuation/case never block a match."""
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _paper_link(paper):
    return paper.get("landing_page_url") or paper.get("doi") or ""


def _year_of(paper):
    year = paper.get("year")
    return year if isinstance(year, int) and not isinstance(year, bool) else None


def _truncate(text, limit=_LABEL_MAX_CHARS):
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "\u2026"


def _plural(count, word):
    """1 -> '1 paper', 4 -> '4 papers'."""
    return f"{count} {word}" if count == 1 else f"{count} {word}s"


def _theme_sort_key(theme_id):
    return (0, int(theme_id)) if str(theme_id).isdigit() else (1, str(theme_id))


# ---------------------------------------------------------------------------
# 1. headline statistics
# ---------------------------------------------------------------------------
def _headline(report):
    papers = report.get("papers") or []
    years = [y for y in (_year_of(p) for p in papers) if y is not None]
    citations = [p["cited_by_count"] for p in papers
                 if isinstance(p.get("cited_by_count"), (int, float))
                 and not isinstance(p.get("cited_by_count"), bool)]
    oa_known = [p for p in papers if "is_oa" in p]
    novelty = report.get("novelty_check") or []

    evaluation = report.get("evaluation") or {}

    def score_of(name):
        part = evaluation.get(name)
        return part.get("score") if isinstance(part, dict) else None

    return {
        "topic": report.get("topic", ""),
        "num_papers": report.get("num_papers", len(papers)),
        "num_themes": len(report.get("themes") or {}),
        "num_hypotheses": len(report.get("hypotheses") or []),
        "critic_approved": bool(report.get("critic_approved")),
        "critic_iterations": report.get("critic_iterations"),
        "critic_support_ratio": report.get("critic_support_ratio"),
        "num_close_matches": sum(1 for n in novelty if n.get("is_close_match")),
        # Quality scores from the Evaluator (None when the run has none).
        "evaluation_overall": (evaluation.get("overall") or {}).get("score"),
        "evaluation_coverage": score_of("coverage"),
        "evaluation_specificity": score_of("specificity"),
        "evaluation_support": score_of("support"),
        "evaluation_novelty": score_of("novelty"),
        "year_min": min(years) if years else None,
        "year_max": max(years) if years else None,
        "avg_citations": round(sum(citations) / len(citations), 1) if citations else None,
        # None (not 0) when the saved data predates the open-access field,
        # so the UI can show "not recorded" instead of a misleading 0%.
        "open_access_pct": (round(100 * sum(1 for p in oa_known if p.get("is_oa")) / len(oa_known))
                            if oa_known else None),
    }


# ---------------------------------------------------------------------------
# 2. shared papers
# ---------------------------------------------------------------------------
def _compare_papers(report_a, report_b):
    papers_a = report_a.get("papers") or []
    papers_b = report_b.get("papers") or []

    dois_b = {d for d in (_normalize_doi(p.get("doi")) for p in papers_b) if d}
    titles_b = {t for t in (_normalize_title(p.get("title")) for p in papers_b) if t}

    shared = []
    shared_keys = set()
    for p in papers_a:
        doi = _normalize_doi(p.get("doi"))
        title = _normalize_title(p.get("title"))
        matched = (bool(doi) and doi in dois_b) or (bool(title) and title in titles_b)
        if not matched:
            continue
        key = ("doi", doi) if doi else ("title", title)
        if key in shared_keys:       # same paper listed twice in A
            continue
        shared_keys.add(key)
        shared.append({
            "title": p.get("title") or "Untitled",
            "year": _year_of(p),
            "link": _paper_link(p),
        })

    # Count unique papers on each side so Jaccard is not distorted by duplicates.
    def unique_keys(papers):
        keys = set()
        for p in papers:
            doi = _normalize_doi(p.get("doi"))
            title = _normalize_title(p.get("title"))
            keys.add(("doi", doi) if doi else ("title", title))
        return keys

    count_a, count_b = len(unique_keys(papers_a)), len(unique_keys(papers_b))
    shared_count = len(shared)
    union = count_a + count_b - shared_count

    return {
        "count_a": count_a,
        "count_b": count_b,
        "shared_count": shared_count,
        "union_count": union,
        "jaccard": round(shared_count / union, 3) if union else 0.0,
        "pct_of_a": round(100 * shared_count / count_a) if count_a else 0,
        "pct_of_b": round(100 * shared_count / count_b) if count_b else 0,
        "shared": shared[:_MAX_SHARED_PAPERS_LISTED],
    }


# ---------------------------------------------------------------------------
# 3. key terms
# ---------------------------------------------------------------------------
def _key_terms_of(report):
    terms = report.get("key_terms")
    if not terms:
        terms = []
        for h in report.get("hypotheses") or []:
            terms.extend(h.get("grounding_terms") or [])
    seen, ordered = set(), []
    for t in terms:
        key = str(t).strip().lower()
        if key and key not in seen:
            seen.add(key)
            ordered.append(str(t).strip())
    return ordered


def _compare_key_terms(report_a, report_b):
    terms_a, terms_b = _key_terms_of(report_a), _key_terms_of(report_b)
    set_a = {t.lower() for t in terms_a}
    set_b = {t.lower() for t in terms_b}
    return {
        "shared": [t for t in terms_a if t.lower() in set_b],
        "only_a": [t for t in terms_a if t.lower() not in set_b],
        "only_b": [t for t in terms_b if t.lower() not in set_a],
    }


# ---------------------------------------------------------------------------
# 4. theme alignment
# ---------------------------------------------------------------------------
def _themes_of(report):
    """Returns (ids, labels, texts, paper_counts) for a report's themes."""
    themes = report.get("themes") or {}
    ids = sorted((str(k) for k in themes), key=_theme_sort_key)
    labels, texts, counts = [], [], []
    for tid in ids:
        theme = themes.get(tid)
        if theme is None:                        # keys may be ints in memory
            theme = themes.get(int(tid)) if tid.isdigit() else {}
        sentences = theme.get("summary_sentences") or []
        titles = theme.get("paper_titles") or []
        text = " ".join(sentences) if sentences else " ".join(titles)
        texts.append(text or " ")
        labels.append(_truncate(sentences[0]) if sentences else _truncate(titles[0] if titles else "(empty theme)"))
        counts.append(theme.get("paper_count", len(titles)))
    return ids, labels, texts, counts


def _unit_rows(matrix):
    matrix = np.asarray(matrix, dtype=float)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _similarity_matrix(texts_a, texts_b, embed_fn):
    """Cosine similarity between every A text and every B text.
    Returns (matrix, method_description)."""
    if not texts_a or not texts_b:
        return np.zeros((len(texts_a), len(texts_b))), "none (a search has no themes)"

    try:
        if embed_fn is None:
            from vector_store import embed_texts   # loaded once by the running app
            embed_fn = embed_texts
            method = f"sentence embeddings ({config.EMBEDDING_MODEL_NAME})"
        else:
            method = "custom embedding function"
        vectors = _unit_rows(embed_fn(list(texts_a) + list(texts_b)))
        if vectors.shape[0] != len(texts_a) + len(texts_b):
            raise ValueError("embedding function returned the wrong number of vectors")
        matrix = vectors[: len(texts_a)] @ vectors[len(texts_a):].T
        return matrix, method
    except Exception as error:
        print(f"[comparison] Embedding model unavailable ({error}); using TF-IDF similarity instead.")

    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidf = TfidfVectorizer(stop_words="english").fit_transform(list(texts_a) + list(texts_b)).toarray()
    vectors = _unit_rows(tfidf)
    return vectors[: len(texts_a)] @ vectors[len(texts_a):].T, "TF-IDF cosine similarity (fallback)"


def _relation(score):
    if score >= THEME_ALIGNED_THRESHOLD:
        return "aligned"
    if score >= THEME_RELATED_THRESHOLD:
        return "related"
    return "distinct"


def _compare_themes(report_a, report_b, embed_fn):
    ids_a, labels_a, texts_a, counts_a = _themes_of(report_a)
    ids_b, labels_b, texts_b, counts_b = _themes_of(report_b)
    matrix, method = _similarity_matrix(texts_a, texts_b, embed_fn)

    alignments = []
    for i, tid in enumerate(ids_a):
        entry = {"a_theme": tid, "a_label": labels_a[i], "a_papers": counts_a[i],
                 "b_theme": None, "b_label": None, "b_papers": None,
                 "similarity": None, "relation": "distinct", "mutual": False}
        if ids_b:
            j = int(np.argmax(matrix[i]))
            score = float(matrix[i, j])
            entry.update({
                "b_theme": ids_b[j], "b_label": labels_b[j], "b_papers": counts_b[j],
                "similarity": round(score, 3), "relation": _relation(score),
                "mutual": int(np.argmax(matrix[:, j])) == i,
            })
        alignments.append(entry)

    unique_b = []
    for j, tid in enumerate(ids_b):
        best = float(matrix[:, j].max()) if ids_a else 0.0
        if _relation(best) == "distinct":
            unique_b.append({"b_theme": tid, "b_label": labels_b[j], "b_papers": counts_b[j],
                             "best_similarity": round(best, 3)})

    return {
        "method": method,
        "thresholds": {"aligned": THEME_ALIGNED_THRESHOLD, "related": THEME_RELATED_THRESHOLD},
        "a_ids": ids_a, "b_ids": ids_b,
        "a_labels": labels_a, "b_labels": labels_b,
        "a_papers": counts_a, "b_papers": counts_b,
        "matrix": [[round(float(v), 3) for v in row] for row in matrix],
        "alignments": alignments,
        "unique_a": [a for a in alignments if a["relation"] == "distinct"],
        "unique_b": unique_b,
    }


# ---------------------------------------------------------------------------
# 5. template-based plain-language summary
# ---------------------------------------------------------------------------
def _build_summary(stats_a, stats_b, papers, terms, themes):
    lines = [
        f"Search A (\u201c{stats_a['topic']}\u201d) analysed {_plural(stats_a['num_papers'], 'paper')} in "
        f"{_plural(stats_a['num_themes'], 'theme')}; search B (\u201c{stats_b['topic']}\u201d) analysed "
        f"{_plural(stats_b['num_papers'], 'paper')} in {_plural(stats_b['num_themes'], 'theme')}."
    ]

    if papers["shared_count"] == 0:
        lines.append("The two searches share no papers, so their evidence bases are independent.")
    else:
        plural = "paper" if papers["shared_count"] == 1 else "papers"
        lines.append(
            f"They share {papers['shared_count']} {plural} "
            f"({papers['pct_of_a']}% of A, {papers['pct_of_b']}% of B; "
            f"{papers['union_count']} distinct papers across both)."
        )

    if terms["shared"]:
        shown = ", ".join(terms["shared"][:5])
        lines.append(f"Key terms in common: {shown}.")
    else:
        lines.append("They have no key terms in common.")

    scored = [a for a in themes["alignments"] if a["similarity"] is not None]
    if scored:
        best = max(scored, key=lambda a: a["similarity"])
        lines.append(
            f"The most similar pair of themes is A{best['a_theme']} and B{best['b_theme']} "
            f"(similarity {best['similarity']}, {best['relation']})."
        )
        n_aligned = sum(1 for a in scored if a["relation"] == "aligned")
        n_related = sum(1 for a in scored if a["relation"] == "related")
        n_unique_a, n_unique_b = len(themes["unique_a"]), len(themes["unique_b"])
        lines.append(
            f"Of the {_plural(len(scored), 'theme')} in A: {n_aligned} aligned, {n_related} related, "
            f"{n_unique_a} unique to A (no close counterpart in B). "
            f"{_plural(n_unique_b, 'theme')} in B {'has' if n_unique_b == 1 else 'have'} no close counterpart in A."
        )

    if stats_a["critic_approved"] == stats_b["critic_approved"]:
        verdict = "were both approved" if stats_a["critic_approved"] else "both ended as best effort"
        lines.append(f"The Critic's verdicts: the two reviews {verdict}.")
    else:
        which = "A" if stats_a["critic_approved"] else "B"
        lines.append(f"Only search {which} was approved by the Critic; the other ended as best effort.")

    quality_a, quality_b = stats_a.get("evaluation_overall"), stats_b.get("evaluation_overall")
    if quality_a is not None and quality_b is not None:
        if abs(quality_a - quality_b) < 0.005:
            lines.append(f"Both searches have the same overall quality score ({quality_a}).")
        else:
            higher = "A" if quality_a > quality_b else "B"
            lines.append(
                f"Search {higher} has the higher overall quality score "
                f"({max(quality_a, quality_b)} against {min(quality_a, quality_b)})."
            )

    return lines


# ---------------------------------------------------------------------------
# plain-English explanation shown at the bottom of the comparison
# ---------------------------------------------------------------------------
def _how_to_read(method):
    """
    Short, jargon-free explanations of how every number on the
    comparison screen was produced. Returned as a list of
    {"term": ..., "text": ...} so the UI can show it as a tidy list.
    """
    if method.startswith("sentence embeddings"):
        similarity_text = (
            "Each theme is a short summary of a group of papers. A language model "
            f"({config.EMBEDDING_MODEL_NAME}) turns every summary into numbers, and each theme "
            "in A is compared with each theme in B. The score runs from 0 (nothing in common) "
            "to 1 (same meaning). A score just below 0, such as -0.03, also simply means "
            "'nothing in common'."
        )
    else:
        similarity_text = (
            "Each theme is a short summary of a group of papers. Each theme in A is compared "
            "with each theme in B by the words they share (TF-IDF). This simpler method was "
            "used because the language model was not available. The score runs from 0 "
            "(no words in common) to 1 (identical wording)."
        )

    return [
        {"term": "Shared papers",
         "text": "Two papers count as the same paper when they have the same DOI (the paper's "
                 "permanent ID number). If a paper has no DOI, its title is used instead, "
                 "ignoring capital letters and punctuation."},
        {"term": "Theme similarity",
         "text": similarity_text},
        {"term": "Aligned, related, distinct",
         "text": f"Aligned: score {THEME_ALIGNED_THRESHOLD} or higher (clearly the same topic). "
                 f"Related: from {THEME_RELATED_THRESHOLD} up to {THEME_ALIGNED_THRESHOLD} "
                 f"(overlapping topic). Distinct: below {THEME_RELATED_THRESHOLD} (different topic)."},
        {"term": "Mutual best",
         "text": "A theme in A and a theme in B are each other's closest match."},
        {"term": "Key terms",
         "text": "Words picked automatically from the papers of each search. \"Shared\" means "
                 "the word was picked for both searches."},
        {"term": "Quality scores",
         "text": "These come from the Evaluator, which scores every search from 0 to 1 (higher is "
                 "better): how well the summary covers the papers, how specific the hypotheses are, "
                 "how much of the summary the Critic found supported, and how many hypotheses have "
                 "no close published match. Open a search to see exactly how each score was worked out."},
        {"term": "Please note",
         "text": f"The scores and labels are a guide, not proof. The cut-off values "
                 f"({THEME_ALIGNED_THRESHOLD} and {THEME_RELATED_THRESHOLD}) were chosen by the "
                 "project, so read the theme text to confirm that two themes really ask the "
                 "same question."},
    ]


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------
def compare_reports(report_a, report_b, embed_fn=None):
    """
    Compares two finished reports (dicts shaped like the ones app.py
    saves to the Search History). Returns a JSON-serialisable dict.
    `embed_fn` is optional and only for testing: a function taking a
    list of strings and returning an array of embeddings. By default
    the project's own embedding model is used.
    """
    stats_a, stats_b = _headline(report_a), _headline(report_b)
    papers = _compare_papers(report_a, report_b)
    terms = _compare_key_terms(report_a, report_b)
    themes = _compare_themes(report_a, report_b, embed_fn)

    return {
        "a": stats_a,
        "b": stats_b,
        "papers": papers,
        "key_terms": terms,
        "themes": themes,
        "summary": _build_summary(stats_a, stats_b, papers, terms, themes),
        "method_note": (
            "Similarity scores and labels are a guide, not proof that two themes address "
            "the same research question."
        ),
        "how_to_read": _how_to_read(themes["method"]),
    }