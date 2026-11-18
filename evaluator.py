"""
evaluator.py
The Evaluator agent: scores the QUALITY of a finished literature-review
run. It looks at the finished report (the same dictionary app.py saves
to the Search History) and produces transparent, explainable scores.
Everything is local and classical - TF-IDF (scikit-learn), string
matching and arithmetic. No LLM and no external API.

What is scored (every score is between 0 and 1)
-----------------------------------------------
1. COVERAGE - does the synthesis actually reflect the retrieved papers?
     paper_representation : share of the retrieved papers that contributed
                            at least one sentence to the theme summaries.
     term_coverage        : the K most characteristic terms of the papers
                            (top TF-IDF terms of their titles + abstracts,
                            excluding the search-topic words, their
                            variants and generic academic words) - how
                            many of them appear in
                            the summaries.
     coverage score       = average of the two.

2. HYPOTHESIS SPECIFICITY - are the hypotheses concrete and testable,
   or generic? Each hypothesis is scored by five weighted checks:
     grounding_quality (0.35) : share of its grounding terms that are real
                                subject words, not generic words such as
                                "big", "good", "best", or bare numbers.
     corpus_grounded   (0.15) : share of its grounding terms that really
                                occur in the retrieved papers (so the
                                hypothesis is anchored in the literature).
     testable_relation (0.20) : phrased as a relationship ("effect of",
                                "moderated by", ...) AND names two or more
                                of its grounding terms (two variables).
     distinct          (0.20) : addresses different variables from the
                                other hypotheses (not a near-duplicate).
     on_topic          (0.10) : mentions a word from the search topic.
   The specificity score is the average over all hypotheses.

3. SUPPORT  - the Critic's final support ratio (how many summary sentences
              the NLI model found supported by the source abstracts).
4. NOVELTY  - share of hypotheses with NO close published match.
5. OVERALL  - the simple, equal-weight average of whichever of the four
              scores above are available.

Honest-framing note: these are heuristic quality indicators that make a
run's strengths and weaknesses visible and comparable; they are not a
guarantee that a hypothesis is correct or scientifically valuable. The
result carries a plain-English "how_scored" explanation so the scoring
can be explained to anyone reading the report.

evaluate() works on any finished report, including ones loaded from the
Search History, so older runs can be scored too.
"""

import re

from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

# --- tunable settings --------------------------------------------------------
TOP_K_TERMS = 15            # how many characteristic terms coverage looks for

LABEL_STRONG = 0.75         # score >= this  -> "strong"
LABEL_FAIR = 0.50           # score >= this  -> "fair", below -> "weak"

COVERAGE_WEIGHTS = {"paper_representation": 0.5, "term_coverage": 0.5}
SPECIFICITY_WEIGHTS = {
    "grounding_quality": 0.35,
    "corpus_grounded": 0.15,
    "testable_relation": 0.20,
    "distinct": 0.20,
    "on_topic": 0.10,
}

# Words that carry no subject-matter meaning in a research hypothesis.
GENERIC_WORDS = {
    "good", "best", "better", "worst", "bad", "big", "large", "small", "high", "higher",
    "low", "lower", "new", "old", "average", "major", "key", "main", "many", "several",
    "various", "different", "important", "significant", "general", "overall", "recent",
    "current", "strong", "early", "late", "long", "short", "full", "real", "true",
    "free", "open", "common", "similar", "specific", "certain", "possible", "potential",
    "number", "level", "way", "case", "fact", "part", "set", "type", "kind", "lot",
    "define", "foundation", "strategies", "understanding",
}

# Words that appear in almost every abstract and say nothing about its subject.
ACADEMIC_STOPWORDS = {
    "abstract", "paper", "papers", "article", "articles", "study", "studies", "research",
    "result", "results", "based", "using", "used", "use", "approach", "approaches",
    "method", "methods", "propose", "proposed", "present", "presents", "presented",
    "show", "shows", "shown", "find", "findings", "found", "review", "analysis",
    "discuss", "discusses", "work", "works", "model", "models", "system", "systems",
    "also", "however", "within", "across", "among", "can", "may", "one", "two",
    "first", "second", "new", "novel", "existing", "provide", "provides", "provided",
    "address", "aim", "aims", "goal", "objective", "purpose", "case", "data",
    "information", "need", "needs", "important", "including", "include", "includes",
    "well", "various", "different", "many", "several", "number", "ieee", "elsevier",
    "springer", "copyright", "rights", "reserved", "doi", "http", "https", "www",
    "order", "field", "area", "areas", "topic", "topics", "issue", "role", "years",
}

# Cues that a sentence states a relationship between things.
_RELATION_CUES = re.compile(
    r"\b(effect|effects|relationship|association|influence|influences|affect|affects|"
    r"moderat\w+|mediat\w+|differ\w*|predict\w*|explain\w*|increase\w*|decrease\w*|"
    r"improve\w*|reduce\w*|correlat\w*|compar\w+|variation|outcomes?|"
    r"will (?:report|clarify|reveal|show|yield))\b",
    re.IGNORECASE,
)

_TOKEN = re.compile(r"[a-z][a-z\-]{2,}")


# --- small helpers ------------------------------------------------------------
def _tokens(text):
    return _TOKEN.findall(str(text or "").lower())


def _fold(word):
    """Very light plural folding so 'systems' and 'system' count as one word."""
    word = word.lower().strip("-")
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _normalize_ws(text):
    return " ".join(str(text or "").lower().split())


def _label(score):
    if score is None:
        return None
    if score >= LABEL_STRONG:
        return "strong"
    if score >= LABEL_FAIR:
        return "fair"
    return "weak"


def _round(value, digits=3):
    return None if value is None else round(float(value), digits)


def _topic_words(topic):
    return {_fold(t) for t in _tokens(topic) if t not in ENGLISH_STOP_WORDS}


# Short acronyms whose expansion is "the topic in other words".
_ACRONYM_EXPANSIONS = {
    "ai": {"artificial", "intelligence"},
    "ml": {"machine", "learning"},
    "dl": {"deep", "learning"},
    "nlp": {"natural", "language", "processing"},
    "iot": {"internet", "things"},
    "llm": {"large", "language", "model"},
    "llms": {"large", "language", "model"},
}


def _topic_exclusions(topic):
    """Words that merely restate the search topic (the topic words and the
    expansion of any acronym in it)."""
    excluded = set()
    for word in re.findall(r"[a-z]+", str(topic or "").lower()):
        if word in ENGLISH_STOP_WORDS:
            continue
        excluded.add(word)
        excluded.add(_fold(word))
        excluded |= _ACRONYM_EXPANSIONS.get(word, set())
    return excluded


def _is_topic_variant(term, excluded):
    """True for the topic words themselves and close variants of them
    (ethical -> ethics, hacking -> hackers): same first four letters."""
    folded = _fold(term)
    if folded in excluded or term.lower() in excluded:
        return True
    for word in excluded:
        if min(len(word), len(folded)) < 4:
            continue
        common = 0
        for a, b in zip(word, folded):
            if a != b:
                break
            common += 1
        if common >= 4:
            return True
    return False


def _paper_text(paper):
    return f"{paper.get('title') or ''}. {paper.get('abstract') or ''}"


def _summary_sentences(report):
    sentences = []
    themes = report.get("themes") or {}
    for key in themes:
        theme = themes[key] or {}
        sentences.extend(s for s in (theme.get("summary_sentences") or []) if s)
    return sentences


# --- characteristic terms of the literature ---------------------------------------
def corpus_key_terms(papers, topic, k=TOP_K_TERMS):
    """
    The k most characteristic single words of the papers: TF-IDF over
    each paper's title + abstract, summed across papers. The search-topic
    words, generic academic words and generic filler words are excluded
    so the list describes what the literature is ABOUT, beyond the topic.
    """
    docs = [_paper_text(p) for p in papers if _paper_text(p).strip(". ")]
    if not docs:
        return []

    stop = set(ENGLISH_STOP_WORDS) | ACADEMIC_STOPWORDS | GENERIC_WORDS
    stop |= set(_tokens(topic)) | _topic_words(topic)

    try:
        vectorizer = TfidfVectorizer(
            stop_words=sorted(stop),
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-]{2,}\b",
            sublinear_tf=True,
            lowercase=True,
        )
        matrix = vectorizer.fit_transform(docs)
    except ValueError:               # nothing left after stop-word removal
        return []

    totals = matrix.sum(axis=0).A1
    names = vectorizer.get_feature_names_out()
    ranked = sorted(zip(names, totals), key=lambda pair: (-pair[1], pair[0]))

    excluded = _topic_exclusions(topic)
    folded_stop = {_fold(w) for w in stop}          # so "issues" is caught by "issue"
    terms, seen = [], set()
    for term, _ in ranked:
        folded = _fold(term)
        if folded in seen or folded in folded_stop or _is_topic_variant(term, excluded):
            continue
        seen.add(folded)
        terms.append(term)
        if len(terms) == k:
            break
    return terms


# --- 1. coverage ---------------------------------------------------------------------
def _sentence_in_abstract(sentence, abstract_norm, abstract_words):
    """True if a summary sentence was taken from this abstract."""
    sent_norm = _normalize_ws(sentence).strip(" .;:")
    if len(sent_norm) < 20:
        return False
    if sent_norm in abstract_norm:
        return True
    words = _tokens(sent_norm)
    if len(words) < 6:
        return False
    # Fallback for light edits (stripped prefixes, changed punctuation):
    # nearly every word of the sentence must occur in the abstract.
    return sum(1 for w in words if w in abstract_words) / len(words) >= 0.9


def score_coverage(report, key_terms):
    papers = report.get("papers") or []
    sentences = _summary_sentences(report)
    if not papers:
        return None

    represented = 0
    for paper in papers:
        abstract_norm = _normalize_ws(paper.get("abstract"))
        abstract_words = set(_tokens(abstract_norm))
        if abstract_norm and any(_sentence_in_abstract(s, abstract_norm, abstract_words) for s in sentences):
            represented += 1
    paper_representation = represented / len(papers)

    summary_words = {_fold(w) for w in _tokens(" ".join(sentences))}
    covered = [t for t in key_terms if _fold(t) in summary_words]
    missing = [t for t in key_terms if _fold(t) not in summary_words]
    term_coverage = (len(covered) / len(key_terms)) if key_terms else None

    parts = {"paper_representation": paper_representation, "term_coverage": term_coverage}
    usable = {k: v for k, v in parts.items() if v is not None}
    weight_sum = sum(COVERAGE_WEIGHTS[k] for k in usable)
    score = sum(COVERAGE_WEIGHTS[k] * v for k, v in usable.items()) / weight_sum

    return {
        "score": _round(score),
        "label": _label(score),
        "paper_representation": _round(paper_representation),
        "papers_represented": represented,
        "papers_total": len(papers),
        "term_coverage": _round(term_coverage),
        "top_terms": key_terms,
        "terms_covered": covered,
        "terms_missing": missing,
    }


# --- 2. hypothesis specificity ---------------------------------------------------------
def _mentions(text_lower, term):
    return re.search(r"\b" + re.escape(str(term).lower()) + r"\b", text_lower) is not None


def _is_substantive(term):
    """A grounding term is substantive if it is made of real subject words."""
    words = _tokens(term)
    raw = str(term).strip()
    if not words or any(ch.isdigit() for ch in raw):
        return False
    return all(len(w) >= 3 and w not in GENERIC_WORDS and _fold(w) not in GENERIC_WORDS for w in words)


def _variables_of(hypothesis, key_terms):
    """The set of 'variable' words a hypothesis is about (for the distinctness check)."""
    folded_key = {_fold(t) for t in key_terms}
    text_words = {_fold(w) for w in _tokens(hypothesis.get("hypothesis"))}
    grounding = {_fold(w) for t in (hypothesis.get("grounding_terms") or []) for w in _tokens(t)}
    return grounding | (text_words & folded_key)


def score_specificity(report, key_terms, corpus_words):
    hypotheses = report.get("hypotheses") or []
    if not hypotheses:
        return None

    topic_words = _topic_words(report.get("topic", ""))
    variable_sets = [_variables_of(h, key_terms) for h in hypotheses]

    per_hypothesis = []
    flagged = []
    for i, h in enumerate(hypotheses):
        text = str(h.get("hypothesis") or "")
        text_lower = text.lower()
        grounding = list(h.get("grounding_terms") or [])
        notes = []

        # 1. grounding quality
        if grounding:
            good_terms = [t for t in grounding if _is_substantive(t)]
            bad_terms = [t for t in grounding if not _is_substantive(t)]
            grounding_quality = len(good_terms) / len(grounding)
            for t in bad_terms:
                flagged.append(t)
            if bad_terms:
                notes.append("generic key term(s): " + ", ".join(bad_terms))
        else:
            grounding_quality = 0.0
            notes.append("no key terms")

        # 2. corpus grounded: do the grounding terms really occur in the papers?
        if grounding:
            found = [t for t in grounding if _tokens(t) and all(_fold(w) in corpus_words for w in _tokens(t))]
            corpus_grounded = len(found) / len(grounding)
            missing_terms = [t for t in grounding if t not in found]
            if missing_terms:
                notes.append("key term(s) not found in the retrieved papers: " + ", ".join(missing_terms))
        else:
            corpus_grounded = 0.0

        # 3. testable relation (cue + two variables mentioned)
        cue = _RELATION_CUES.search(text) is not None
        mentioned = [t for t in grounding if _mentions(text_lower, t)]
        if cue and len(set(m.lower() for m in mentioned)) >= 2:
            testable_relation = 1.0
        elif cue:
            testable_relation = 0.5
            notes.append("states a relationship but names fewer than two of its key terms")
        else:
            testable_relation = 0.0
            notes.append("does not state a relationship between variables")

        # 4. distinct from the other hypotheses
        others = [v for j, v in enumerate(variable_sets) if j != i]
        mine = variable_sets[i]
        if not mine:
            distinct = 0.0
            notes.append("no identifiable variables")
        elif not others:
            distinct = 1.0
        else:
            worst = max(len(mine & o) / len(mine | o) for o in others if (mine | o))
            distinct = 1.0 - worst
            if worst >= 0.6:
                notes.append("very similar variables to another hypothesis")

        # 5. on topic
        hyp_words = {_fold(w) for w in _tokens(text)}
        on_topic = 1.0 if (not topic_words or (topic_words & hyp_words)) else 0.0
        if not on_topic:
            notes.append("does not mention the search topic")

        checks = {
            "grounding_quality": _round(grounding_quality),
            "corpus_grounded": _round(corpus_grounded),
            "testable_relation": _round(testable_relation),
            "distinct": _round(distinct),
            "on_topic": _round(on_topic),
        }
        score = sum(SPECIFICITY_WEIGHTS[k] * v for k, v in
                    {"grounding_quality": grounding_quality, "corpus_grounded": corpus_grounded,
                     "testable_relation": testable_relation, "distinct": distinct,
                     "on_topic": on_topic}.items())
        per_hypothesis.append({
            "id": h.get("id", f"H{i + 1}"),
            "score": _round(score),
            "label": _label(score),
            "checks": checks,
            "notes": notes,
        })

    mean = sum(p["score"] for p in per_hypothesis) / len(per_hypothesis)
    seen, flagged_unique = set(), []
    for t in flagged:
        if t.lower() not in seen:
            seen.add(t.lower())
            flagged_unique.append(t)

    return {
        "score": _round(mean),
        "label": _label(mean),
        "per_hypothesis": per_hypothesis,
        "flagged_terms": flagged_unique,
    }


# --- 3 & 4. support and novelty -----------------------------------------------------------
def score_support(report):
    ratio = report.get("critic_support_ratio")
    if not isinstance(ratio, (int, float)) or isinstance(ratio, bool):
        return None
    return {"score": _round(ratio), "label": _label(ratio), "approved": bool(report.get("critic_approved"))}


def score_novelty(report):
    checked = [n for n in (report.get("novelty_check") or []) if n.get("checked")]
    if not checked:
        return None
    close = sum(1 for n in checked if n.get("is_close_match"))
    score = 1 - close / len(checked)
    return {"score": _round(score), "label": _label(score), "checked": len(checked), "close_matches": close}


# --- plain-English explanation ---------------------------------------------------------------
def _how_scored():
    return [
        {"term": "Coverage",
         "text": "Does the summary really reflect the papers? Half of the score is the share of papers "
                 "that contributed at least one sentence to a theme summary. The other half is how many "
                 f"of the {TOP_K_TERMS} most characteristic words of the papers (found automatically with "
                 "TF-IDF, ignoring the search topic itself) appear in the summaries."},
        {"term": "Hypothesis specificity",
         "text": "Is each hypothesis concrete and testable, or generic? Every hypothesis is checked for: "
                 "subject words instead of generic ones like \"good\" or \"big\" (35%), words that are "
                 "found in the retrieved papers (15%), a stated relationship between two variables (20%), "
                 "different variables from the other hypotheses (20%), and mention of the topic (10%). "
                 "The score is the average over all hypotheses."},
        {"term": "Support",
         "text": "The Critic's final support ratio: the share of summary sentences that the language-"
                 "inference model judged to be backed up by the source abstracts."},
        {"term": "Novelty",
         "text": "The share of hypotheses for which no closely matching published paper was found. "
                 "\"No close match found\" does not prove originality."},
        {"term": "Overall",
         "text": "A simple average of the scores above, each counted equally. Strong means 0.75 or higher, "
                 "fair means 0.50 to 0.75, weak means below 0.50."},
        {"term": "Please note",
         "text": "These are quick quality indicators that make weak spots easy to see and runs easy to "
                 "compare. They do not prove that a hypothesis is correct or scientifically valuable."},
    ]


# --- plain-language weaknesses -------------------------------------------------------------------
def _weaknesses(coverage, specificity, novelty, support):
    notes = []
    if coverage:
        if coverage["paper_representation"] is not None and coverage["paper_representation"] < 0.6:
            notes.append(f"Only {coverage['papers_represented']} of {coverage['papers_total']} papers "
                         "contributed a sentence to the theme summaries.")
        if coverage["term_coverage"] is not None and coverage["term_coverage"] < 0.5:
            missing = ", ".join(coverage["terms_missing"][:5])
            notes.append(f"The summaries miss {len(coverage['terms_missing'])} of the "
                         f"{len(coverage['top_terms'])} most characteristic terms (for example: {missing}).")
    if specificity:
        if specificity["flagged_terms"]:
            notes.append("Some hypotheses are built around generic key words: " + ", ".join(specificity["flagged_terms"]) + ".")
        weak = [p["id"] for p in specificity["per_hypothesis"] if p["score"] < LABEL_FAIR]
        if weak:
            notes.append("Weak hypotheses: " + ", ".join(weak) + ".")
        similar = [p["id"] for p in specificity["per_hypothesis"] if p["checks"]["distinct"] is not None and p["checks"]["distinct"] < 0.4]
        if similar:
            notes.append("Very similar to another hypothesis: " + ", ".join(similar) + ".")
    if novelty and novelty["close_matches"]:
        noun, verb = ("hypothesis", "shows") if novelty["checked"] == 1 else ("hypotheses", "show")
        notes.append(f"{novelty['close_matches']} of {novelty['checked']} {noun} {verb} possible overlap "
                     "with published work.")
    if support and support["score"] < LABEL_STRONG:
        notes.append(f"The Critic found only {int(round(support['score'] * 100))}% of summary sentences supported.")
    return notes


# --- public entry point --------------------------------------------------------------------------
def evaluate(report):
    """
    Scores a finished report (a dict shaped like the ones app.py saves
    to the Search History). Returns a JSON-serialisable dict. Missing
    parts are tolerated: a score that cannot be computed is None and is
    simply left out of the overall average.
    """
    papers = report.get("papers") or []
    key_terms = corpus_key_terms(papers, report.get("topic", ""))

    coverage = score_coverage(report, key_terms)
    corpus_words = {_fold(w) for p in papers for w in _tokens(_paper_text(p))}
    specificity = score_specificity(report, key_terms, corpus_words)
    support = score_support(report)
    novelty = score_novelty(report)

    components = {
        "coverage": coverage["score"] if coverage else None,
        "specificity": specificity["score"] if specificity else None,
        "support": support["score"] if support else None,
        "novelty": novelty["score"] if novelty else None,
    }
    used = {k: v for k, v in components.items() if v is not None}
    overall_score = sum(used.values()) / len(used) if used else None

    return {
        "coverage": coverage,
        "specificity": specificity,
        "support": support,
        "novelty": novelty,
        "overall": {
            "score": _round(overall_score),
            "label": _label(overall_score),
            "components_used": sorted(used),
        },
        "weaknesses": _weaknesses(coverage, specificity, novelty, support),
        "how_scored": _how_scored(),
    }


if __name__ == "__main__":
    # Quick manual check: score the most recent saved search, if any.
    import history

    records = history.list_history()
    if not records:
        print("No saved searches yet - run a search in the app first.")
    else:
        entry = history.load_run(records[0]["id"])
        result = evaluate(entry["report"])
        print(f"Topic: {entry['topic']}")
        print(f"Overall: {result['overall']['score']} ({result['overall']['label']})")
        for name in ("coverage", "specificity", "support", "novelty"):
            part = result[name]
            print(f"  {name}: {part['score'] if part else 'n/a'} ({part['label'] if part else ''})")
        for line in result["weaknesses"]:
            print("  -", line)