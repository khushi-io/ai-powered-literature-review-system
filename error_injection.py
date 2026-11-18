"""
error_injection.py
Generates deliberately corrupted versions of genuine synthesis
sentences, so critic.py's fact-checking ability can be properly
tested. Without this, testing the Critic against extractive
sentences (which are copied verbatim from source abstracts) is
trivial - it will always find perfect support, since the sentence
IS the source. Error injection creates a real pass/fail test: a
working Critic should approve genuine sentences and flag corrupted
ones. This follows the synthetic error-injection approach used in
factual-consistency research (e.g. FactCC, SummaC-style evaluation).
"""

import re
import random

NEGATION_WORDS = ["not", "does not", "cannot", "fails to", "no longer"]

# Common comparative/impact words we can flip to their rough opposite,
# to simulate a "reversed finding" type of error.
OPPOSITE_PAIRS = [
    ("increases", "decreases"), ("improves", "worsens"),
    ("outperforms", "underperforms"), ("higher", "lower"),
    ("positive", "negative"), ("significant", "negligible"),
    ("effective", "ineffective"), ("accurate", "inaccurate"),
    ("supports", "contradicts"), ("more", "less"),
]

NUMBER_PATTERN = re.compile(r'\b\d+(\.\d+)?\b')


def inject_negation(sentence):
    """
    Flips the meaning of a sentence by inserting a negation before
    the first verb-like word, or swapping a known opposite pair.
    Returns None if no suitable insertion point was found.
    """
    lower = sentence.lower()
    for word_a, word_b in OPPOSITE_PAIRS:
        if word_a in lower:
            idx = lower.find(word_a)
            corrupted = sentence[:idx] + word_b + sentence[idx + len(word_a):]
            return corrupted
        if word_b in lower:
            idx = lower.find(word_b)
            corrupted = sentence[:idx] + word_a + sentence[idx + len(word_b):]
            return corrupted
    return None


def inject_number_swap(sentence, replacement_pool):
    """
    Replaces a number in the sentence with a different number found
    elsewhere in the dataset (replacement_pool = list of other
    sentences to pull numbers from). Returns None if no number
    exists in the sentence, or no replacement was found.
    """
    numbers_in_sentence = NUMBER_PATTERN.findall(sentence)
    match = NUMBER_PATTERN.search(sentence)
    if not match:
        return None

    original_number = match.group()
    candidates = []
    for other in replacement_pool:
        for other_match in NUMBER_PATTERN.finditer(other):
            candidate = other_match.group()
            if candidate != original_number:
                candidates.append(candidate)

    if not candidates:
        return None

    new_number = random.choice(candidates)
    corrupted = sentence[:match.start()] + new_number + sentence[match.end():]
    return corrupted


def inject_splice(sentence, other_sentences):
    """
    Replaces the second half of `sentence` with the second half of a
    randomly chosen, unrelated sentence - creates a fluent-sounding
    but factually spliced/false claim. Returns None if not enough
    material to splice with.
    """
    candidates = [s for s in other_sentences if s != sentence and len(s.split()) > 6]
    if not candidates:
        return None

    other = random.choice(candidates)

    words_a = sentence.split()
    words_b = other.split()
    midpoint_a = len(words_a) // 2
    midpoint_b = len(words_b) // 2

    spliced = " ".join(words_a[:midpoint_a] + words_b[midpoint_b:])
    return spliced


def generate_corrupted_variant(sentence, all_sentences_pool):
    """
    Tries negation first (cleanest, most realistic error type), then
    falls back to number swap, then splice. Returns a dict describing
    what was done, or None if no corruption technique applied.
    """
    other_sentences = [s for s in all_sentences_pool if s != sentence]

    negated = inject_negation(sentence)
    if negated:
        return {"original": sentence, "corrupted": negated, "technique": "negation"}

    number_swapped = inject_number_swap(sentence, other_sentences)
    if number_swapped:
        return {"original": sentence, "corrupted": number_swapped, "technique": "number_swap"}

    spliced = inject_splice(sentence, other_sentences)
    if spliced:
        return {"original": sentence, "corrupted": spliced, "technique": "splice"}

    return None


def build_test_set(genuine_sentences):
    """
    Given a list of genuine (real, source-supported) sentences,
    builds a labeled test set: each genuine sentence paired with one
    corrupted variant (where possible). Returns a list of dicts:
    {"sentence": ..., "is_genuine": True/False, "technique": ... or None}

    This is what critic.py's evaluation harness will run through the
    NLI checker to measure how well it discriminates real vs corrupted.
    """
    test_set = []
    for sentence in genuine_sentences:
        test_set.append({"sentence": sentence, "is_genuine": True, "technique": None})

        variant = generate_corrupted_variant(sentence, genuine_sentences)
        if variant:
            test_set.append({
                "sentence": variant["corrupted"],
                "is_genuine": False,
                "technique": variant["technique"],
            })

    return test_set


if __name__ == "__main__":
    # Quick manual test
    sample_sentences = [
        "Machine learning improves crop yield prediction accuracy significantly.",
        "The study used 50 papers from six electronic databases for analysis.",
        "Soil moisture variables were found to be most influential on the ML predictions.",
        "Temperature and rainfall are the most commonly used features in yield models.",
    ]

    test_set = build_test_set(sample_sentences)
    for item in test_set:
        label = "GENUINE" if item["is_genuine"] else f"CORRUPTED ({item['technique']})"
        print(f"[{label}] {item['sentence']}")