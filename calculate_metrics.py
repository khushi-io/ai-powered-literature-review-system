"""
calculate_metrics.py
Runs the core research experiment described in the project synopsis
(Section 8). Since the Synthesizer is purely extractive (its
sentences are copied verbatim from source abstracts), a genuine,
unmodified synthesis is already ~100% self-supported - there is
nothing for a refinement loop to catch. To create a fair, meaningful
test of the Critic's value, this script deliberately corrupts a
portion of each synthesis's sentences (via error_injection.py) to
simulate the kind of unsupported claims an UNVERIFIED single-pass
system could produce, then compares:

  Condition A (baseline): the error-containing synthesis, checked
    ONCE with no revision - represents a system with no verification
    step, where errors go uncaught.
  Condition B (treatment): the SAME error-containing synthesis, run
    through the full Critic self-refinement loop - represents this
    project's proposed system, where errors are detected and removed.

The same retrieved paper set and the same injected corruption are
used for both conditions per topic, so the only difference is
whether the Critic loop was applied - a controlled ablation.

Usage:
    python calculate_metrics.py
"""

import os
import csv
import json
import copy
import time
import random

import config
from retriever import retrieve_papers
from synthesizer import synthesize
from critic import critique_synthesis, _revise_synthesis, run_discrimination_test
import error_injection

random.seed(42)  # reproducible corruption pattern across runs

# ---------------------------------------------------------------------------
# Test topic set - edit this list to match what you want to report on.
# Aim for ~10-20 topics across a couple of different fields, per the
# synopsis's feasibility check (Section 8.6).
# ---------------------------------------------------------------------------
TEST_TOPICS = [
    "machine learning for crop yield prediction",
    "deep learning for medical image segmentation",
    "natural language processing for sentiment analysis",
    "reinforcement learning for robotics control",
    "convolutional neural networks for image classification",
    "machine learning for network intrusion detection",
    "deep learning for speech recognition",
    "transfer learning for low resource languages",
    "graph neural networks for recommendation systems",
    "federated learning for privacy preserving machine learning",
]

CORRUPTION_RATE = 0.4  # fraction of synthesis sentences deliberately corrupted per topic

RESULTS_CSV_PATH = os.path.join(config.BASE_DIR, "experiment_results.csv")
SUMMARY_JSON_PATH = os.path.join(config.BASE_DIR, "experiment_summary.json")


def _build_corrupted_synthesis(synthesis, corruption_rate=CORRUPTION_RATE):
    """
    Returns a deep-copied synthesis where a fraction of the summary
    sentences in each theme have been replaced with deliberately
    corrupted variants (via error_injection.py), simulating the kind
    of unsupported claim an unverified single-pass system could
    produce. Sentences that couldn't be corrupted (no suitable
    technique found) are left as-is.
    """
    corrupted = copy.deepcopy(synthesis)

    all_sentences = []
    for theme in corrupted["themes"].values():
        all_sentences.extend(theme["summary_sentences"])

    for theme in corrupted["themes"].values():
        new_sentences = []
        for sentence in theme["summary_sentences"]:
            if random.random() < corruption_rate:
                variant = error_injection.generate_corrupted_variant(sentence, all_sentences)
                if variant:
                    new_sentences.append(variant["corrupted"])
                    continue
            new_sentences.append(sentence)
        theme["summary_sentences"] = new_sentences

    return corrupted


def run_condition_a(corrupted_synthesis, papers):
    """
    Condition A (Baseline): the error-containing synthesis, checked
    ONCE against sources, with NO revision loop. Represents a system
    with no verification step - errors go uncaught in the final output.
    """
    synthesis_copy = copy.deepcopy(corrupted_synthesis)
    verdict = critique_synthesis(synthesis_copy, papers)
    return {
        "support_ratio": verdict["support_ratio"],
        "total_sentences": verdict["total_sentences"],
        "supported_sentences": verdict["supported_sentences"],
    }


def run_condition_b(corrupted_synthesis, papers, approval_threshold=0.7):
    """
    Condition B (Treatment): the SAME error-containing synthesis, run
    through the Critic's verify -> revise -> re-verify loop, up to
    config.MAX_CRITIC_REFINEMENT_LOOPS iterations. Unsupported
    (including corrupted) sentences get dropped each iteration.
    Runs entirely in-memory - does NOT use critic.py's disk cache,
    since that cache is keyed by topic and must not be overwritten
    with results computed on a deliberately corrupted synthesis
    (that cache is also used by pipeline.py for real demo runs).
    """
    synthesis = copy.deepcopy(corrupted_synthesis)
    verdict = None
    iterations_used = 0

    for iteration in range(1, config.MAX_CRITIC_REFINEMENT_LOOPS + 1):
        verdict = critique_synthesis(synthesis, papers)
        iterations_used = iteration

        if verdict["support_ratio"] >= approval_threshold:
            return {
                "support_ratio": verdict["support_ratio"],
                "total_sentences": verdict["total_sentences"],
                "supported_sentences": verdict["supported_sentences"],
                "approved": True,
                "iterations_used": iterations_used,
            }

        if iteration < config.MAX_CRITIC_REFINEMENT_LOOPS:
            synthesis = _revise_synthesis(synthesis, verdict, papers)

    return {
        "support_ratio": verdict["support_ratio"],
        "total_sentences": verdict["total_sentences"],
        "supported_sentences": verdict["supported_sentences"],
        "approved": False,
        "iterations_used": iterations_used,
    }


def run_experiment(topics=None, force_refresh_retrieval=False):
    if topics is None:
        topics = TEST_TOPICS

    all_rows = []

    for i, topic in enumerate(topics, 1):
        print(f"\n{'=' * 70}")
        print(f"[{i}/{len(topics)}] TOPIC: {topic}")
        print("=" * 70)

        try:
            papers = retrieve_papers(topic, force_refresh=force_refresh_retrieval)
            if len(papers) < 2:
                print(f"[calculate_metrics] SKIPPING - not enough papers with abstracts ({len(papers)} found).")
                continue

            real_synthesis = synthesize(topic)
            corrupted_synthesis = _build_corrupted_synthesis(real_synthesis)

            print("[calculate_metrics] Running Condition A (baseline, no critic, errors uncaught)...")
            t0 = time.time()
            condition_a = run_condition_a(corrupted_synthesis, papers)
            time_a = round(time.time() - t0, 2)
            print(f"  Condition A support ratio: {condition_a['support_ratio']} "
                  f"({condition_a['supported_sentences']}/{condition_a['total_sentences']}) in {time_a}s")

            print("[calculate_metrics] Running Condition B (Critic self-refinement loop)...")
            t0 = time.time()
            condition_b = run_condition_b(corrupted_synthesis, papers)
            time_b = round(time.time() - t0, 2)
            print(f"  Condition B support ratio: {condition_b['support_ratio']} "
                  f"({condition_b['supported_sentences']}/{condition_b['total_sentences']}), "
                  f"iterations: {condition_b['iterations_used']}, approved: {condition_b['approved']} in {time_b}s")

            print("[calculate_metrics] Running discrimination test (genuine vs corrupted)...")
            disc = run_discrimination_test(topic, force_refresh=False)
            print(f"  Discrimination accuracy: {disc['accuracy']}, "
                  f"precision: {disc['precision_detecting_corrupted']}, "
                  f"recall: {disc['recall_detecting_corrupted']}, "
                  f"f1: {disc['f1_detecting_corrupted']}")

            improvement = round(condition_b["support_ratio"] - condition_a["support_ratio"], 3)

            all_rows.append({
                "topic": topic,
                "num_papers": len(papers),
                "condition_a_support_ratio": condition_a["support_ratio"],
                "condition_b_support_ratio": condition_b["support_ratio"],
                "improvement": improvement,
                "condition_b_iterations_used": condition_b["iterations_used"],
                "condition_b_approved": condition_b["approved"],
                "discrimination_accuracy": disc["accuracy"],
                "discrimination_precision": disc["precision_detecting_corrupted"],
                "discrimination_recall": disc["recall_detecting_corrupted"],
                "discrimination_f1": disc["f1_detecting_corrupted"],
            })

        except Exception as e:
            print(f"[calculate_metrics] ERROR on topic '{topic}': {e}")
            continue

    return all_rows


def save_results_csv(rows, path=RESULTS_CSV_PATH):
    if not rows:
        print("[calculate_metrics] No results to save.")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[calculate_metrics] Per-topic results saved to {path}")


def compute_summary(rows):
    if not rows:
        return {}

    n = len(rows)
    mean_a = sum(r["condition_a_support_ratio"] for r in rows) / n
    mean_b = sum(r["condition_b_support_ratio"] for r in rows) / n
    mean_improvement = sum(r["improvement"] for r in rows) / n
    topics_improved = sum(1 for r in rows if r["improvement"] > 0)
    topics_converged_within_max = sum(
        1 for r in rows if r["condition_b_iterations_used"] <= config.MAX_CRITIC_REFINEMENT_LOOPS
    )
    mean_iterations = sum(r["condition_b_iterations_used"] for r in rows) / n
    mean_disc_accuracy = sum(r["discrimination_accuracy"] for r in rows) / n
    mean_disc_precision = sum(r["discrimination_precision"] for r in rows) / n
    mean_disc_recall = sum(r["discrimination_recall"] for r in rows) / n
    mean_disc_f1 = sum(r["discrimination_f1"] for r in rows) / n

    summary = {
        "num_topics_tested": n,
        "corruption_rate_used": CORRUPTION_RATE,
        "mean_condition_a_support_ratio": round(mean_a, 3),
        "mean_condition_b_support_ratio": round(mean_b, 3),
        "mean_improvement_b_over_a": round(mean_improvement, 3),
        "topics_where_b_improved_over_a": topics_improved,
        "topics_converged_within_max_iterations": topics_converged_within_max,
        "mean_iterations_used": round(mean_iterations, 2),
        "mean_discrimination_accuracy": round(mean_disc_accuracy, 3),
        "mean_discrimination_precision": round(mean_disc_precision, 3),
        "mean_discrimination_recall": round(mean_disc_recall, 3),
        "mean_discrimination_f1": round(mean_disc_f1, 3),
    }
    return summary


def print_summary(summary):
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY (H1 / H2 EVIDENCE)")
    print("=" * 70)
    print(f"Topics tested: {summary['num_topics_tested']}  "
          f"(synthetic corruption rate: {summary['corruption_rate_used']})")
    print(f"\nMean Condition A (baseline, errors uncaught) support ratio: {summary['mean_condition_a_support_ratio']}")
    print(f"Mean Condition B (Critic loop) support ratio: {summary['mean_condition_b_support_ratio']}")
    print(f"Mean improvement (B - A): {summary['mean_improvement_b_over_a']}")
    print(f"Topics where B improved over A: {summary['topics_where_b_improved_over_a']} / {summary['num_topics_tested']}")
    print(f"\nMean iterations used by Critic loop: {summary['mean_iterations_used']}")
    print(f"Topics converged within {config.MAX_CRITIC_REFINEMENT_LOOPS} iterations: "
          f"{summary['topics_converged_within_max_iterations']} / {summary['num_topics_tested']}")
    print(f"\nMean discrimination test accuracy: {summary['mean_discrimination_accuracy']}")
    print(f"Mean discrimination test precision: {summary['mean_discrimination_precision']}")
    print(f"Mean discrimination test recall: {summary['mean_discrimination_recall']}")
    print(f"Mean discrimination test F1: {summary['mean_discrimination_f1']}")
    print("=" * 70)


if __name__ == "__main__":
    print(f"Running experiment across {len(TEST_TOPICS)} topics...")
    print("This will take a while on first run (live OpenAlex calls + NLI checks per topic).\n")

    results = run_experiment()
    save_results_csv(results)

    summary = compute_summary(results)
    with open(SUMMARY_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[calculate_metrics] Summary saved to {SUMMARY_JSON_PATH}")

    print_summary(summary)