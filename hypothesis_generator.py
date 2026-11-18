"""
hypothesis_generator.py
Hypothesis Generator agent: drafts testable research hypotheses from
a Critic-approved literature synthesis - no external LLM API. Uses
TF-IDF (fit on the FULL paper corpus, so word distinctiveness is
measured against real background frequency) to extract key terms
from gap sentences and low-coverage themes. Filters applied to keep
terms meaningful:
  (1) exclude any word already present in the topic string itself,
      including basic plural normalization and short words (e.g. "AI");
  (2) exclude common ABBREVIATIONS of the topic's own words (e.g.
      topic "Machine Learning..." also excludes "ml");
  (3) exclude generic academic filler words that aren't meaningful
      "gap" content on their own;
  (4) prefer two-word phrases (bigrams, e.g. "deep learning") over
      single words, and once a bigram is chosen, skip any single
      word that is just one of its component words;
  (5) verify a candidate bigram is a REAL, contiguous phrase in the
      source text before trusting it - scikit-learn builds bigrams
      AFTER removing stopwords, so two words that were never actually
      adjacent (e.g. "...diagnosis on a massive scale...") can end up
      glued together as a fake phrase ("diagnosis massive"). Filter
      (5) rejects any bigram that doesn't literally appear as a
      contiguous phrase in the real text, falling back to single
      words instead.
Remaining terms are slotted into structured hypothesis templates.

Accepts an optional `settings` dict (see config.DEFAULT_SETTINGS) -
`settings["num_hypotheses"]` controls how many hypotheses are
drafted, and settings are also passed through to retriever for the
background corpus and used for cache-key naming. The internal
retrieve_papers call always uses force_refresh=False: by the time
this stage runs, the Retriever's own cache has already been written
(freshly, if a live re-run was requested at the top level), so
re-forcing a live OpenAlex fetch here would only be redundant.
"""

import os
import json
import random

from sklearn.feature_extraction.text import TfidfVectorizer

import config
from retriever import _store_name, retrieve_papers

random.seed(42)  # reproducible template/term selection for demos

HYPOTHESIS_TEMPLATES = [
    "The effect of {term1} on {term2} in the context of {topic} remains insufficiently studied; "
    "it is hypothesized that {term1} has a measurable, quantifiable influence on {term2} outcomes.",

    "Given limited research coverage on {term1} within {topic}, it is hypothesized that incorporating "
    "{term1} alongside {term2} will yield improved predictive or explanatory performance compared to "
    "approaches that rely on {term2} alone.",

    "It is hypothesized that the relationship between {term1} and {term2} in {topic} is moderated by "
    "factors not yet addressed in existing literature, and that controlling for these factors will "
    "reveal a stronger or different association than currently reported.",

    "Because current studies on {topic} show limited consensus regarding {term1}, it is hypothesized "
    "that a controlled comparative study isolating {term1} from {term2} will clarify its independent "
    "contribution.",

    "It is hypothesized that {term1}, considered jointly with {term2}, explains variation in outcomes "
    "for {topic} that neither factor accounts for when examined in isolation.",

    "Given the comparatively sparse treatment of {term1} in the existing {topic} literature, it is "
    "hypothesized that studies explicitly incorporating {term1} will report different conclusions "
    "than studies that omit it.",
]

# Generic academic/methods vocabulary that shows up in almost any
# paper regardless of subject - not useful as a "gap" term even
# though it may score as statistically distinctive in a small sample.
ACADEMIC_STOPWORDS = {
    "using", "used", "use", "based", "method", "methods", "approach",
    "approaches", "model", "models", "study", "studies", "analysis",
    "data", "results", "result", "feature", "features", "algorithm",
    "algorithms", "performance", "research", "paper", "propose",
    "proposed", "technique", "techniques", "review", "literature",
    "systematic", "article", "articles", "work", "different", "various",
    "previously", "previous", "existing", "recent", "recently", "novel",
    "significant", "significantly", "effective", "effectively",
    "surprisingly", "understudied", "current", "currently", "field",
    "problem", "problems", "solution", "solutions", "address",
    "addressing", "important", "widely", "large", "well", "new",
}

# Common abbreviations mapped to their spelled-out form. If ALL words
# of the spelled-out form appear in the topic string, the abbreviation
# is excluded too.
COMMON_ABBREVIATIONS = {
    "ml": "machine learning",
    "ai": "artificial intelligence",
    "dl": "deep learning",
    "nlp": "natural language processing",
    "cv": "computer vision",
    "rl": "reinforcement learning",
    "iot": "internet of things",
    "cnn": "convolutional neural network",
    "rnn": "recurrent neural network",
    "gan": "generative adversarial network",
    "svm": "support vector machine",
}


def _cache_path(topic, settings):
    return os.path.join(config.HYPOTHESIS_CACHE_DIR, f"{_store_name(topic, settings)}.json")


def _normalize(word):
    """Very basic stemming: strip a trailing 's' so 'yields' matches
    'yield', 'models' matches 'model', etc."""
    word = word.lower()
    if word.endswith("s") and len(word) > 3:
        return word[:-1]
    return word


def _topic_word_set(topic):
    """
    Normalized words in the topic itself, to exclude from 'key
    terms', PLUS any common abbreviation whose spelled-out form is
    fully contained in the topic. Includes short words too (e.g. "AI").
    """
    topic_lower = topic.lower()
    topic_words = set(_normalize(w) for w in topic.split() if len(w) >= 2)

    for abbrev, expansion in COMMON_ABBREVIATIONS.items():
        expansion_words = expansion.split()
        if all(w in topic_lower for w in expansion_words):
            topic_words.add(abbrev)

    return topic_words


def _passes_filters(term, excluded_topic_words):
    """True if `term` is safe to use as a key term - i.e. it doesn't
    overlap the topic's own words/abbreviations or generic academic
    filler vocabulary."""
    term_words = set(_normalize(w) for w in term.split())
    if term_words & excluded_topic_words:
        return False
    if term_words & ACADEMIC_STOPWORDS:
        return False
    return True


def _extract_key_terms(background_corpus, target_texts, topic, top_n=8):
    """
    Fits TF-IDF on the FULL background corpus (all paper abstracts),
    so word distinctiveness is measured against real corpus-wide
    frequency, not just the small gap-sentence sample. Then scores
    the target_texts (gap sentences) against that vocabulary.

    Selection prefers two-word phrases (bigrams) over single words,
    but ONLY bigrams that are verified to appear as a real,
    contiguous phrase in the actual source text - scikit-learn
    builds bigrams after removing stopwords, so two words that were
    never actually adjacent can otherwise end up glued together as
    a fake "phrase". Once a genuine bigram is chosen, any single
    word that is one of its component words is skipped, so "deep"
    never appears separately once "deep learning" has already been
    selected. Remaining slots are filled with the highest-scoring
    single words not already covered by a chosen bigram. Returns the
    top_n most distinctive, non-generic, non-fragmented terms.
    """
    if not background_corpus or not target_texts:
        return []

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=200,
        min_df=1,
    )
    try:
        vectorizer.fit(background_corpus)
        target_matrix = vectorizer.transform(target_texts)
    except ValueError:
        return []

    scores = target_matrix.sum(axis=0).A1
    terms = vectorizer.get_feature_names_out()
    excluded_topic_words = _topic_word_set(topic)

    ranked = sorted(zip(terms, scores), key=lambda x: x[1], reverse=True)
    ranked = [(term, score) for term, score in ranked if score > 0]

    # A bigram is only trusted if it appears as a real, contiguous
    # phrase in the actual target text - not just that both words
    # happen to appear somewhere in it.
    combined_text_lower = " ".join(target_texts).lower()

    def _is_real_phrase(term):
        return term.lower() in combined_text_lower

    bigrams = [
        term for term, score in ranked
        if " " in term and _passes_filters(term, excluded_topic_words) and _is_real_phrase(term)
    ]
    unigrams = [term for term, score in ranked if " " not in term and _passes_filters(term, excluded_topic_words)]

    key_terms = []
    covered_words = set()

    # Pass 1: verified bigrams first, skipping any that overlap a
    # bigram already chosen
    for term in bigrams:
        term_words = set(_normalize(w) for w in term.split())
        if term_words & covered_words:
            continue
        key_terms.append(term)
        covered_words |= term_words
        if len(key_terms) >= top_n:
            return key_terms

    # Pass 2: fill remaining slots with single words not already
    # covered by a chosen bigram's component words
    for term in unigrams:
        term_words = set(_normalize(w) for w in term.split())
        if term_words & covered_words:
            continue
        key_terms.append(term)
        covered_words |= term_words
        if len(key_terms) >= top_n:
            break

    return key_terms


def _find_low_coverage_themes(synthesis, max_papers=2):
    """
    Returns summary text from themes with few supporting papers -
    these represent understudied angles, a strong signal for gaps.
    """
    low_coverage_texts = []
    for theme_id, theme in synthesis["themes"].items():
        if theme["paper_count"] <= max_papers:
            low_coverage_texts.extend(theme["summary_sentences"])
    return low_coverage_texts


def generate_hypotheses(critic_result, settings=None, force_refresh=False):
    """
    Main entry point. Takes the Critic's approved result (from
    critic.run_critic_loop) and generates testable hypotheses from
    its gap signals and low-coverage themes. The number of
    hypotheses drafted is controlled by settings["num_hypotheses"].
    """
    topic = critic_result["topic"]
    merged = config.merge_settings(settings)
    cache_file = _cache_path(topic, merged)

    if os.path.exists(cache_file) and not force_refresh:
        print(f"[hypothesis_generator] Cache hit for '{topic}'")
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)

    synthesis = critic_result["approved_synthesis"]

    gap_texts = synthesis.get("gaps", [])
    low_coverage_texts = _find_low_coverage_themes(synthesis)
    combined_gap_material = gap_texts + low_coverage_texts

    if not combined_gap_material:
        for theme in synthesis["themes"].values():
            combined_gap_material.extend(theme["summary_sentences"])

    # force_refresh is deliberately NOT passed through here - by the
    # time this stage runs, the Retriever's own cache already holds
    # the correct paper set (freshly fetched at the top level if a
    # live re-run was requested), so forcing another live fetch here
    # would only be a redundant, slower repeat of the same call.
    papers = retrieve_papers(topic, settings=merged, force_refresh=False)
    background_corpus = [p["abstract"] for p in papers]

    key_terms = _extract_key_terms(background_corpus, combined_gap_material, topic, top_n=8)

    if len(key_terms) < 2:
        print("[hypothesis_generator] WARNING: too few distinctive key terms extracted; "
              "hypotheses may be generic. Consider a broader topic or more papers.")
        key_terms = key_terms + ["the identified factors", "reported outcomes"]

    hypotheses = []
    num_to_generate = min(merged["num_hypotheses"], len(HYPOTHESIS_TEMPLATES))
    templates_to_use = random.sample(HYPOTHESIS_TEMPLATES, num_to_generate)

    for i, template in enumerate(templates_to_use):
        term1 = key_terms[i % len(key_terms)]
        term2 = key_terms[(i + 1) % len(key_terms)]
        hypothesis_text = template.format(term1=term1, term2=term2, topic=topic)
        hypotheses.append({
            "id": f"H{i + 1}",
            "hypothesis": hypothesis_text,
            "grounding_terms": [term1, term2],
        })

    result = {
        "topic": topic,
        "critic_approved": critic_result["approved"],
        "key_terms_extracted": key_terms,
        "hypotheses": hypotheses,
    }

    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"[hypothesis_generator] Cached {len(hypotheses)} hypotheses to {cache_file}")

    return result


if __name__ == "__main__":
    from synthesizer import synthesize
    from critic import run_critic_loop

    topic = "machine learning for crop yield prediction"
    synthesis = synthesize(topic)
    critic_result = run_critic_loop(synthesis, topic)
    hyp_result = generate_hypotheses(critic_result, force_refresh=True)

    print("\n=== GENERATED HYPOTHESES ===")
    print(f"Key terms extracted: {hyp_result['key_terms_extracted']}")
    print()
    for h in hyp_result["hypotheses"]:
        print(f"[{h['id']}] {h['hypothesis']}\n")