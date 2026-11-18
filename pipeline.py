"""
pipeline.py
Command-line entry point for the full pipeline:
Retriever -> Synthesizer -> Critic -> Hypothesis Generator ->
Novelty Checker -> Evaluator.

The agents are wired together by a LangGraph state graph (see
graph_pipeline.py); this file only reads the command-line options,
prints progress while the graph runs, and prints the final report.

Usage:
    python pipeline.py "your research topic here"
    python pipeline.py "your research topic here" --fresh

Settings Panel flags (all optional - defaults come from
config.DEFAULT_SETTINGS if omitted):
    --max-papers N        papers KEPT after dedup/relevance filtering (3-30)
    --fetch-limit N       raw papers PULLED from OpenAlex before filtering (5-50)
    --min-relevance F     0.0-0.6, 0.0 = no relevance filtering at all
    --year-from YYYY      earliest publication year to include
    --year-to YYYY        latest publication year to include
    --recent-only         shortcut: only papers from the last 3 years
    --open-access         only include open-access papers
    --num-hypotheses N    how many hypotheses to draft (1-6)
    --skip-discrimination skip the (slow) genuine-vs-corrupted claim test

Example:
    python pipeline.py "machine learning in potato yield production" --fresh --min-relevance 0 --max-papers 10

    python pipeline.py --diagram      prints the pipeline graph as Mermaid text
"""

import sys
import json
import argparse

import graph_pipeline

STAGE_TITLES = {
    "retriever": "STAGE 1: RETRIEVER \u2014 fetching papers from OpenAlex",
    "synthesizer": "STAGE 2: SYNTHESIZER \u2014 clustering into themes + TextRank summarization",
    "critic": "STAGE 3: CRITIC \u2014 NLI-based fact verification + self-refinement loop",
    "discrimination": "STAGE 3b: CRITIC DISCRIMINATION TEST \u2014 genuine vs corrupted claim detection",
    "hypothesis": "STAGE 4: HYPOTHESIS GENERATOR \u2014 TF-IDF key terms + template drafting",
    "novelty": "STAGE 5: NOVELTY CHECKER \u2014 comparing hypotheses against published work",
    "evaluator": "STAGE 6: EVALUATOR \u2014 coverage and hypothesis-specificity scoring",
}


def _parse_args():
    parser = argparse.ArgumentParser(description="Run the literature review pipeline on a topic.")
    parser.add_argument("topic", nargs="*", help="Research topic (quote it, or pass multiple words)")
    parser.add_argument("--fresh", action="store_true", help="Bypass cache, force a live re-run")
    parser.add_argument("--max-papers", type=int, default=None)
    parser.add_argument("--fetch-limit", type=int, default=None)
    parser.add_argument("--min-relevance", type=float, default=None)
    parser.add_argument("--year-from", type=int, default=None)
    parser.add_argument("--year-to", type=int, default=None)
    parser.add_argument("--recent-only", action="store_true", default=None)
    parser.add_argument("--open-access", action="store_true", default=None)
    parser.add_argument("--num-hypotheses", type=int, default=None)
    parser.add_argument("--skip-discrimination", action="store_true",
                        help="Skip the slow genuine-vs-corrupted claim detection test")
    parser.add_argument("--diagram", action="store_true",
                        help="Print the pipeline graph as Mermaid text and exit")
    args = parser.parse_args()

    topic = " ".join(args.topic).strip() if args.topic else None

    settings = {
        "max_papers": args.max_papers,
        "fetch_limit": args.fetch_limit,
        "min_relevance": args.min_relevance,
        "year_from": args.year_from,
        "year_to": args.year_to,
        "recent_only": args.recent_only,
        "open_access_only": args.open_access,
        "num_hypotheses": args.num_hypotheses,
    }
    # Drop keys the user didn't set, so config.merge_settings falls
    # back to defaults for them instead of overwriting with None.
    settings = {k: v for k, v in settings.items() if v is not None}

    return topic, args.fresh, settings, (not args.skip_discrimination), args.diagram


def _print_header(text):
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


def _print_progress(stage, status, message):
    """Progress callback handed to the graph: prints a header when a stage
    starts and its one-line result when it finishes."""
    if status == "running":
        _print_header(STAGE_TITLES.get(stage, stage.upper()))
    elif status == "done":
        print(f"-> {message}")
    elif status == "error":
        print(f"-> ERROR: {message}")


def run_pipeline(topic, settings=None, force_fresh=False, run_discrimination=True):
    """
    Runs the full pipeline and returns the final report dictionary
    (or None if the run stopped early, for example because too few
    papers were found).
    """
    state = graph_pipeline.run_graph(
        topic,
        settings=settings,
        force_fresh=force_fresh,
        run_discrimination=run_discrimination,
        progress=_print_progress,
    )

    if state.get("error"):
        print("\n" + state["error"])
        return None

    report = state["final_report"]
    _print_final_report(report)
    return report


def _print_final_report(report):
    _print_header("FINAL REPORT")
    print(f"Topic: {report['topic']}")
    print(f"Papers analyzed: {report['num_papers']}")
    print(f"Themes identified: {report['num_themes']}")
    print(f"Critic verdict: {'APPROVED' if report['critic_approved'] else 'NOT APPROVED'} "
          f"(iterations used: {report['critic_iterations']}, "
          f"support ratio: {report['critic_support_ratio']})")

    if report.get("discrimination_test"):
        d = report["discrimination_test"]
        print(f"Critic discrimination test: accuracy={d['accuracy']}, "
              f"precision={d['precision_detecting_corrupted']}, "
              f"recall={d['recall_detecting_corrupted']}, f1={d['f1_detecting_corrupted']}")

    print("\n--- THEMES ---")
    for theme_id, theme in report["themes"].items():
        print(f"\nTheme {theme_id} ({theme['paper_count']} papers):")
        for s in theme["summary_sentences"]:
            print(f"  - {s}")

    if report.get("gaps"):
        print("\n--- IDENTIFIED GAPS ---")
        for g in report["gaps"]:
            print(f"  - {g}")

    print("\n--- GENERATED HYPOTHESES ---")
    novelty_by_id = {n["hypothesis_id"]: n for n in report.get("novelty_check", [])}
    for h in report["hypotheses"]:
        print(f"\n[{h['id']}] {h['hypothesis']}")
        n = novelty_by_id.get(h["id"])
        if n and n.get("checked") and n.get("best_match_title"):
            print(f"  Novelty check: similarity {n['similarity']} to \"{n['best_match_title']}\" ({n.get('best_match_year')})")
            print(f"  {n['verdict']}")

    evaluation = report.get("evaluation")
    print("\n--- EVALUATION ---")
    if not evaluation or evaluation["overall"]["score"] is None:
        print("  (no evaluation available for this run)")
    else:
        overall = evaluation["overall"]
        print(f"  Overall: {overall['score']} ({overall['label']})")
        for name in ("coverage", "specificity", "support", "novelty"):
            part = evaluation.get(name)
            if part:
                print(f"  {name.capitalize():<12} {part['score']} ({part['label']})")
        for note in evaluation["weaknesses"]:
            print(f"  ! {note}")

    print("\n--- TIMINGS ---")
    for stage, seconds in report["timings_seconds"].items():
        print(f"  {stage}: {seconds}s")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    topic, force_fresh, settings, run_discrimination, show_diagram = _parse_args()

    if show_diagram:
        print(graph_pipeline.graph_as_mermaid())
        sys.exit(0)

    if not topic:
        topic = input("Enter a research topic: ").strip()

    if not topic:
        print("No topic provided. Exiting.")
        sys.exit(1)

    result = run_pipeline(topic, settings=settings, force_fresh=force_fresh,
                          run_discrimination=run_discrimination)

    if result:
        output_file = f"final_report_{topic[:40].replace(' ', '_')}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\nFull JSON report saved to: {output_file}")