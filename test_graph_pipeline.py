"""
test_graph_pipeline.py
Self-test for graph_pipeline.py (the LangGraph pipeline).

The graph is the REAL one (real LangGraph, real routing, real Evaluator);
only the heavy agents (OpenAlex retrieval, language models) are replaced
by tiny stand-ins, so every check is fast and its outcome is predictable.
The final check prints the pipeline diagram as Mermaid text.

Run with:  python test_graph_pipeline.py
Nothing is written to disk and no internet or model is needed.
"""

import copy
import json

import graph_pipeline as gp

checks_passed = 0
checks_failed = 0


def check(name, condition, extra=""):
    global checks_passed, checks_failed
    if condition:
        checks_passed += 1
        print(f"  PASS  {name}")
    else:
        checks_failed += 1
        print(f"  FAIL  {name}" + (f"   [{extra}]" if extra else ""))


# ---------------------------------------------------------------------------
# stand-in agents + a recorder that remembers how they were called
# ---------------------------------------------------------------------------
S1 = "Firewalls inspect packets and block suspicious connections before they reach internal servers."
S2 = "Intrusion detection systems raise alerts when traffic patterns match known malware signatures."
S_REJECTED = "Quantum computers will make every existing cipher obsolete within a decade."

PAPERS = [
    {"title": "Firewall design", "abstract": S1, "doi": "10.1/a", "year": 2024},
    {"title": "Detecting intrusions", "abstract": S2, "doi": "10.1/b", "year": 2023},
    {"title": "Password safety", "abstract": "Encryption protects stored credentials.", "doi": "10.1/c", "year": 2022},
]


def make_agents(papers=None, calls=None, critic_drops_sentence=True, evaluate=None):
    calls = calls if calls is not None else []
    papers = PAPERS if papers is None else papers

    def retrieve(topic, settings=None, force_refresh=False):
        calls.append(("retrieve", topic, settings, force_refresh))
        return copy.deepcopy(papers)

    def synthesize(topic, settings=None, force_refresh=False):
        calls.append(("synthesize", topic, settings, force_refresh))
        return {"topic": topic, "num_papers": len(papers),
                "themes": {"0": {"paper_titles": ["x"], "paper_count": 3,
                                 "summary_sentences": [S1, S2, S_REJECTED]}},
                "gaps": ["raw gap"], "contradictions": []}

    def critic(synthesis, topic, settings=None, force_refresh=False):
        calls.append(("critic", topic, settings, force_refresh))
        approved = copy.deepcopy(synthesis)
        if critic_drops_sentence:      # the Critic removes the unsupported sentence
            approved["themes"]["0"]["summary_sentences"] = [S1, S2]
        approved["gaps"] = ["approved gap"]
        return {"topic": topic, "approved": True, "iterations_used": 2,
                "final_verdict": {"support_ratio": 1.0}, "approved_synthesis": approved}

    def discrimination(topic, settings=None, force_refresh=False):
        calls.append(("discrimination", topic, settings, force_refresh))
        return {"accuracy": 0.925}

    def hypotheses(critic_result, settings=None, force_refresh=False):
        calls.append(("hypotheses", settings, force_refresh))
        return {"hypotheses": [{"id": "H1", "grounding_terms": ["firewalls", "intrusion"],
                                "hypothesis": "The effect of firewalls on intrusion in the context of "
                                              "network defense remains insufficiently studied."}],
                "key_terms_extracted": ["firewalls", "intrusion"]}

    def novelty(hyp_result, settings=None, force_refresh=False):
        calls.append(("novelty", settings, force_refresh))
        return {"num_hypotheses_checked": 1, "num_close_matches_found": 0,
                "per_hypothesis": [{"hypothesis_id": "H1", "checked": True, "is_close_match": False}]}

    agents = {"retrieve": retrieve, "synthesize": synthesize, "critic": critic,
              "discrimination": discrimination, "hypotheses": hypotheses, "novelty": novelty}
    if evaluate is not None:
        agents["evaluate"] = evaluate
    return agents, calls


def run(topic="network defense", settings=None, force_fresh=False, run_discrimination=False, **agent_kwargs):
    events = []
    agents, calls = make_agents(**agent_kwargs)
    state = gp.run_graph(topic, settings=settings, force_fresh=force_fresh,
                         run_discrimination=run_discrimination,
                         progress=lambda stage, status, message: events.append((stage, status, message)),
                         agents=agents)
    return state, events, calls


# ===========================================================================
print("1. Happy path")
state, events, calls = run()
report = state["final_report"]
check("the run produces a final report and no error", "final_report" in state and not state.get("error"))
check("agents ran in pipeline order", [c[0] for c in calls] == ["retrieve", "synthesize", "critic", "hypotheses", "novelty"], calls)
stages_done = [s for s, status, _ in events if status == "done"]
check("progress reports every stage as done, in order", stages_done == gp.STAGES, stages_done)
check("every stage reports 'running' before 'done'",
      all(events.index((s, "running", next(m for st, sta, m in events if st == s and sta == "running"))) <
          events.index((s, "done", next(m for st, sta, m in events if st == s and sta == "done"))) for s in gp.STAGES))
check("the discrimination test did not run (not requested)", "discrimination" not in [c[0] for c in calls])
check("report has the fields the app, history and export rely on", all(k in report for k in [
    "topic", "settings", "num_papers", "papers", "num_themes", "themes", "gaps", "contradictions",
    "critic_approved", "critic_iterations", "critic_support_ratio", "hypotheses", "key_terms",
    "novelty_check", "evaluation", "timings_seconds"]))
check("report numbers are right", report["num_papers"] == 3 and report["num_themes"] == 1
      and report["critic_iterations"] == 2 and report["critic_support_ratio"] == 1.0)
check("discrimination_test is None when not run", report["discrimination_test"] is None)
check("report is JSON-serialisable", bool(json.dumps(report)))

print("\n2. The report uses the Critic's APPROVED synthesis")
sentences = report["themes"]["0"]["summary_sentences"]
check("the sentence the Critic removed is NOT in the report", S_REJECTED not in sentences and sentences == [S1, S2], sentences)
check("gaps come from the approved synthesis too", report["gaps"] == ["approved gap"])
keep, _, _ = run(critic_drops_sentence=False)
check("if the Critic removes nothing, all sentences are kept", len(keep["final_report"]["themes"]["0"]["summary_sentences"]) == 3)

print("\n3. Evaluator is part of the run")
evaluation = report["evaluation"]
check("the report carries an evaluation", evaluation is not None and evaluation["overall"]["score"] is not None)
check("it has all four component scores", all(evaluation[k] is not None for k in ("coverage", "specificity", "support", "novelty")))
check("it scored the APPROVED text (2 of 3 papers are represented)", evaluation["coverage"]["papers_represented"] == 2, evaluation["coverage"])
last_message = [m for s, status, m in events if s == "evaluator" and status == "done"][0]
check("the evaluator stage reports the overall score", last_message.startswith("Overall "), last_message)

print("\n4. Timings")
timings = report["timings_seconds"]
check("every agent has a timing plus a total", {"retriever", "synthesizer", "critic", "hypothesis_generator",
      "novelty_checker", "evaluator", "total"} <= set(timings), timings)
check("total is at least as long as the slowest single stage", timings["total"] >= max(v for k, v in timings.items() if k != "total"))

print("\n5. Settings and options are passed to every agent")
custom = {"max_papers": 7, "open_access_only": True}
state, _, calls = run(settings=custom, force_fresh=True)
check("the settings are merged with the defaults", state["final_report"]["settings"]["max_papers"] == 7
      and state["final_report"]["settings"]["open_access_only"] is True
      and "min_relevance" in state["final_report"]["settings"])
check("every agent received the same merged settings", all(
    (c[2] if c[0] in ("retrieve", "synthesize", "critic") else c[1]) == state["final_report"]["settings"] for c in calls))
check("force_fresh reaches every agent", all((c[3] if c[0] in ("retrieve", "synthesize", "critic") else c[2]) is True for c in calls))

print("\n6. Routing: too few papers stops the run cleanly")
state, events, calls = run(papers=[PAPERS[0]])
check("no later agent was called", [c[0] for c in calls] == ["retrieve"], calls)
check("the state carries the friendly error message", state.get("error") == gp.NOT_ENOUGH_PAPERS_MESSAGE)
check("there is no final report", "final_report" not in state)
check("the retriever stage is reported as an error", ("retriever", "error", "Not enough papers found for this topic/settings.") in events)
check("no other stage was started", {s for s, _, _ in events} == {"retriever"})
state, _, calls = run(papers=[])
check("zero papers stops the run too", state.get("error") and [c[0] for c in calls] == ["retrieve"])
state, _, calls = run(papers=PAPERS[:2])
check("exactly MIN_PAPERS papers is enough to continue", "final_report" in state)

print("\n7. Routing: the optional discrimination test")
state, events, calls = run(run_discrimination=True)
check("it runs between the Critic and the Hypothesis Generator",
      [c[0] for c in calls] == ["retrieve", "synthesize", "critic", "discrimination", "hypotheses", "novelty"], calls)
check("its result is stored in the report", state["final_report"]["discrimination_test"] == {"accuracy": 0.925})
check("its timing is recorded", "critic_discrimination_test" in state["final_report"]["timings_seconds"])

print("\n8. Failures")
def failing_critic(synthesis, topic, settings=None, force_refresh=False):
    raise RuntimeError("critic exploded")
events = []
agents, calls = make_agents()
agents["critic"] = failing_critic
try:
    gp.run_graph("network defense", progress=lambda s, st, m: events.append((s, st, m)), agents=agents)
    raised = None
except RuntimeError as e:
    raised = str(e)
check("an agent failure is raised to the caller", raised == "critic exploded", raised)
check("the failing stage is reported as an error", ("critic", "error", "critic exploded") in events)
check("later stages never started", not any(s in ("hypothesis", "novelty", "evaluator") for s, _, _ in events))

def failing_evaluate(report):
    raise ValueError("scoring bug")
print("        (a Python error message appears below ON PURPOSE - the test breaks the scorer deliberately)")
state, events, _ = run(evaluate=failing_evaluate)
check("if scoring fails, the finished run is NOT lost", "final_report" in state and state["final_report"]["evaluation"] is None)
check("the evaluator stage still ends as done, with an explanation",
      any(s == "evaluator" and st == "done" and "unavailable" in m for s, st, m in events))

print("\n9. Other behaviour")
agents, _ = make_agents()
quiet = gp.run_graph("network defense", agents=agents)       # no progress callback at all
check("works without a progress callback", "final_report" in quiet)
check("the stage list matches the graph's progress stages", gp.STAGES == ["retriever", "synthesizer", "critic", "hypothesis", "novelty", "evaluator"])
a, _, _ = run(); b, _, _ = run()
check("two identical runs give the same scores (deterministic)", a["final_report"]["evaluation"] == b["final_report"]["evaluation"])

print("\n10. The pipeline diagram")
mermaid = gp.graph_as_mermaid()
check("diagram text is produced", isinstance(mermaid, str) and "graph TD" in mermaid)
check("it mentions every node", all(name in mermaid for name in
      ["retriever", "synthesizer", "critic", "discrimination", "hypothesis", "novelty", "evaluator"]))
print("\n----- Mermaid diagram of the pipeline (paste into mermaid.live to see it) -----")
print(mermaid)
print("---------------------------------------------------------------------------------")

print("\n" + "=" * 70)
print(f"RESULT: {checks_passed} passed, {checks_failed} failed")
if checks_failed == 0:
    print("All graph_pipeline.py checks passed.")