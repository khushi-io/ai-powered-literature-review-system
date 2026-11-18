"""
critic.py
Critic agent: fact-checks the Synthesizer's output against the
original source abstracts using a local NLI (Natural Language
Inference) model - no external LLM API. For each summary sentence,
it first uses embeddings (fast) to shortlist the top-K most relevant
source sentences, then runs the NLI model (slower, more expensive)
only on that shortlist - a standard retrieve-then-verify approach
that avoids checking every sentence against every source sentence
(which is far too slow on CPU). If too many sentences are unsupported,
the weakest sentence is dropped and the synthesis is re-checked, up
to config.MAX_CRITIC_REFINEMENT_LOOPS times.

Also includes a discrimination-test harness (using error_injection.py)
that measures whether the NLI check can actually tell genuine
sentences apart from deliberately corrupted ones.

Accepts an optional `settings` dict (see config.DEFAULT_SETTINGS),
passed through to retriever.retrieve_papers and used for cache-key
naming, so the Critic always checks against the same paper set the
researcher's chosen settings would produce. Internal retrieve_papers
calls always use force_refresh=False: by the time this stage runs,
the Retriever's own cache has already been written (freshly, if a
live re-run was requested at the top level), so re-forcing a live
OpenAlex fetch here would only be a redundant, slower repeat of the
same call.
"""

import os
import json
import numpy as np
from tqdm import tqdm

from sentence_transformers import CrossEncoder

import config
from retriever import retrieve_papers, _store_name
from textrank import split_into_sentences
from vector_store import embed_texts, embed_text
import error_injection

print("Loading NLI model (first time may take a moment)...")
_nli_model = CrossEncoder(config.NLI_MODEL_NAME)
print("NLI model loaded.")

# This model's output order is: [contradiction, entailment, neutral]
LABELS = ["contradiction", "entailment", "neutral"]

TOP_K_EVIDENCE = 5  # how many source sentences to NLI-check per claim


def _cache_path(topic, settings):
    return os.path.join(config.CRITIC_CACHE_DIR, f"{_store_name(topic, settings)}.json")


def _discrimination_cache_path(topic, settings):
    return os.path.join(config.CRITIC_CACHE_DIR, f"{_store_name(topic, settings)}_discrimination_test.json")


def _shortlist_evidence(sentence, source_sentences, source_embeddings, top_k=TOP_K_EVIDENCE):
    """
    Fast pre-filter step: uses embeddings (already computed) to find
    the top_k source sentences most similar to `sentence`, so the
    slower NLI model only has to check a handful of candidates
    instead of every source sentence.
    """
    if len(source_sentences) == 0:
        return []

    query_vec = embed_text(sentence)
    query_vec = query_vec / np.linalg.norm(query_vec)
    norm_embeddings = source_embeddings / np.linalg.norm(source_embeddings, axis=1, keepdims=True)
    scores = norm_embeddings @ query_vec

    top_k = min(top_k, len(source_sentences))
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [source_sentences[i] for i in top_indices]


def _check_sentence_support(sentence, source_sentences, source_embeddings):
    """
    Checks a single synthesis sentence against a SHORTLIST of the
    most relevant source sentences (found via embeddings first).
    Returns the label ('entailment' / 'contradiction' / 'neutral')
    and score for the BEST matching source sentence found.
    """
    shortlist = _shortlist_evidence(sentence, source_sentences, source_embeddings)
    if not shortlist:
        return "neutral", 0.0, None

    pairs = [(src, sentence) for src in shortlist]
    scores = _nli_model.predict(pairs)  # shape (N, 3): [contradiction, entailment, neutral]

    best_entailment_idx = scores[:, 1].argmax()
    best_entailment_score = float(scores[best_entailment_idx][1])

    best_contradiction_idx = scores[:, 0].argmax()
    best_contradiction_score = float(scores[best_contradiction_idx][0])

    if best_entailment_score >= config.NLI_CONTRADICTION_THRESHOLD:
        return "entailment", best_entailment_score, shortlist[best_entailment_idx]
    elif best_contradiction_score >= config.NLI_CONTRADICTION_THRESHOLD:
        return "contradiction", best_contradiction_score, shortlist[best_contradiction_idx]
    else:
        return "neutral", best_entailment_score, None


def _get_all_source_sentences(papers):
    """Flatten all paper abstracts into a single list of sentences."""
    all_sentences = []
    for p in papers:
        all_sentences.extend(split_into_sentences(p["abstract"]))
    return all_sentences


def critique_synthesis(synthesis, papers):
    """
    Checks every summary sentence in every theme against the source
    abstracts (via embedding shortlist + NLI verification). Returns
    a verdict dict with per-sentence labels and an overall support
    ratio for the whole synthesis. Operates on already-retrieved
    `papers`, so it needs no settings of its own.
    """
    source_sentences = _get_all_source_sentences(papers)
    print(f"[critic] Embedding {len(source_sentences)} source sentences once...")
    source_embeddings = embed_texts(source_sentences)

    all_claims = []
    for theme_id, theme in synthesis["themes"].items():
        for sentence in theme["summary_sentences"]:
            all_claims.append((theme_id, sentence))

    sentence_verdicts = []
    for theme_id, sentence in tqdm(all_claims, desc="[critic] Verifying claims"):
        label, score, evidence = _check_sentence_support(sentence, source_sentences, source_embeddings)
        sentence_verdicts.append({
            "theme_id": theme_id,
            "sentence": sentence,
            "label": label,
            "score": round(score, 3),
            "evidence": evidence,
        })

    total = len(sentence_verdicts)
    supported = sum(1 for v in sentence_verdicts if v["label"] == "entailment")
    support_ratio = supported / total if total > 0 else 0.0

    return {
        "sentence_verdicts": sentence_verdicts,
        "total_sentences": total,
        "supported_sentences": supported,
        "support_ratio": round(support_ratio, 3),
    }


def _revise_synthesis(synthesis, verdict, papers):
    """
    Simple revision strategy (no LLM available): for every sentence
    that was NOT entailment-supported, remove it from its theme.
    This shrinks the synthesis down to only well-supported claims.
    """
    unsupported = {
        (v["theme_id"], v["sentence"])
        for v in verdict["sentence_verdicts"]
        if v["label"] != "entailment"
    }

    for theme_id, theme in synthesis["themes"].items():
        theme["summary_sentences"] = [
            s for s in theme["summary_sentences"]
            if (theme_id, s) not in unsupported
        ]

    return synthesis


def run_critic_loop(synthesis, topic, settings=None, approval_threshold=None, force_refresh=False):
    """
    Main entry point: runs the verify -> revise -> re-verify loop,
    up to config.MAX_CRITIC_REFINEMENT_LOOPS times, or until the
    synthesis's support_ratio meets approval_threshold. `settings`
    is used both to fetch the matching paper set and to key the
    cache file, so results from different settings never collide.
    """
    merged = config.merge_settings(settings)
    if approval_threshold is None:
        approval_threshold = config.CRITIC_APPROVAL_THRESHOLD

    cache_file = _cache_path(topic, merged)
    if os.path.exists(cache_file) and not force_refresh:
        print(f"[critic] Cache hit for '{topic}'")
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)

    # Deliberately not forcing a re-fetch here - see the module
    # docstring above for why.
    papers = retrieve_papers(topic, settings=merged, force_refresh=False)

    iteration_log = []
    for iteration in range(1, config.MAX_CRITIC_REFINEMENT_LOOPS + 1):
        print(f"[critic] Iteration {iteration}: checking synthesis against sources...")
        verdict = critique_synthesis(synthesis, papers)
        print(f"[critic] Support ratio: {verdict['support_ratio']} "
              f"({verdict['supported_sentences']}/{verdict['total_sentences']} sentences supported)")

        iteration_log.append({
            "iteration": iteration,
            "support_ratio": verdict["support_ratio"],
            "supported_sentences": verdict["supported_sentences"],
            "total_sentences": verdict["total_sentences"],
        })

        if verdict["support_ratio"] >= approval_threshold:
            print(f"[critic] APPROVED at iteration {iteration}.")
            result = {
                "topic": topic,
                "approved": True,
                "iterations_used": iteration,
                "final_verdict": verdict,
                "iteration_log": iteration_log,
                "approved_synthesis": synthesis,
            }
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            return result

        if iteration < config.MAX_CRITIC_REFINEMENT_LOOPS:
            print(f"[critic] Not yet approved. Revising synthesis (removing unsupported sentences)...")
            synthesis = _revise_synthesis(synthesis, verdict, papers)

    print(f"[critic] Max iterations reached without meeting threshold. Returning best-effort synthesis.")
    result = {
        "topic": topic,
        "approved": False,
        "iterations_used": config.MAX_CRITIC_REFINEMENT_LOOPS,
        "final_verdict": verdict,
        "iteration_log": iteration_log,
        "approved_synthesis": synthesis,
    }
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return result


def run_discrimination_test(topic, settings=None, force_refresh=False):
    """
    THE KEY EVALUATION: measures whether the NLI check can actually
    tell genuine synthesis sentences apart from deliberately
    corrupted ones (via error_injection.py). Reports accuracy,
    precision, recall, and F1 for detecting corrupted sentences.
    """
    merged = config.merge_settings(settings)
    cache_file = _discrimination_cache_path(topic, merged)
    if os.path.exists(cache_file) and not force_refresh:
        print(f"[critic] Discrimination test cache hit for '{topic}'")
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)

    # Deliberately not forcing a re-fetch here - see the module
    # docstring above for why.
    papers = retrieve_papers(topic, settings=merged, force_refresh=False)
    source_sentences = _get_all_source_sentences(papers)

    genuine_sentences = source_sentences[:20]  # cap for speed
    test_set = error_injection.build_test_set(genuine_sentences)

    # Embed the full source pool ONCE, reuse for every test item.
    source_embeddings = embed_texts(source_sentences)

    results = []
    true_positives = 0   # corrupted, correctly flagged as NOT entailment
    false_negatives = 0  # corrupted, incorrectly labeled entailment (missed)
    true_negatives = 0   # genuine, correctly labeled entailment
    false_positives = 0  # genuine, incorrectly flagged as NOT entailment

    for item in tqdm(test_set, desc="[critic] Discrimination test"):
        label, score, evidence = _check_sentence_support(item["sentence"], source_sentences, source_embeddings)

        predicted_genuine = (label == "entailment")
        correct = (predicted_genuine == item["is_genuine"])

        if not item["is_genuine"]:
            if not predicted_genuine:
                true_positives += 1
            else:
                false_negatives += 1
        else:
            if predicted_genuine:
                true_negatives += 1
            else:
                false_positives += 1

        results.append({
            "sentence": item["sentence"],
            "is_genuine": item["is_genuine"],
            "technique": item["technique"],
            "predicted_label": label,
            "score": round(score, 3),
            "evidence_matched": evidence,
            "correct": correct,
        })

    total = len(results)
    accuracy = (true_positives + true_negatives) / total if total > 0 else 0.0
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    summary = {
        "topic": topic,
        "total_test_items": total,
        "accuracy": round(accuracy, 3),
        "precision_detecting_corrupted": round(precision, 3),
        "recall_detecting_corrupted": round(recall, 3),
        "f1_detecting_corrupted": round(f1, 3),
        "true_positives": true_positives,
        "false_negatives": false_negatives,
        "true_negatives": true_negatives,
        "false_positives": false_positives,
        "details": results,
    }

    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[critic] Discrimination test cached to {cache_file}")

    return summary


if __name__ == "__main__":
    from synthesizer import synthesize

    topic = "machine learning for crop yield prediction"

    print("\n=== PART 1: Critic refinement loop on real synthesis ===")
    synthesis = synthesize(topic)
    result = run_critic_loop(synthesis, topic, force_refresh=True)
    print(f"Approved: {result['approved']}")
    print(f"Iterations used: {result['iterations_used']}")
    print(json.dumps(result["iteration_log"], indent=2))

    print("\n=== PART 2: Discrimination test (genuine vs corrupted) ===")
    disc_result = run_discrimination_test(topic, force_refresh=True)
    print(f"Accuracy: {disc_result['accuracy']}")
    print(f"Precision (detecting corrupted): {disc_result['precision_detecting_corrupted']}")
    print(f"Recall (detecting corrupted): {disc_result['recall_detecting_corrupted']}")
    print(f"F1 (detecting corrupted): {disc_result['f1_detecting_corrupted']}")
    print(f"TP={disc_result['true_positives']} FN={disc_result['false_negatives']} "
          f"TN={disc_result['true_negatives']} FP={disc_result['false_positives']}")