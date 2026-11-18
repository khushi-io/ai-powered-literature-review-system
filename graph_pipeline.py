"""
graph_pipeline.py
The multi-agent pipeline, expressed as a LangGraph state graph.

    START
      |
  [retriever] --(fewer than MIN_PAPERS papers)--------------------> END
      |
  [synthesizer]
      |
  [critic] --(run_discrimination switched on)--> [discrimination]
      |                                               |
      +-----------------------------------------------+
      |
  [hypothesis]
      |
  [novelty]
      |
  [evaluator]
      |
    END

Each box is one agent. The graph carries a single shared "state"
dictionary from node to node; every node reads what earlier nodes
produced and adds its own result. Two decisions are made by the graph
itself (conditional edges):

  1. After the Retriever: if too few papers were found the run stops
     cleanly with a helpful message instead of failing later.
  2. After the Critic: the optional discrimination test (genuine vs
     corrupted claim detection, used for the project's evaluation
     figures) runs only when it was asked for.

The Evaluator is the last node. It scores the finished report (see
evaluator.py) and attaches the scores to it, so every run - and every
entry in the Search History - carries its own quality scores.

Callers (the web app and the command line) get live progress by passing
a function  progress(stage, status, message)  - status is one of
"running", "done" or "error". The graph itself does not print anything.

No LLM and no external service is used here: LangGraph is only the
orchestration layer that wires the existing local agents together.
"""

import time
from typing import Any, Callable, Dict, List, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

import config
import evaluator

MIN_PAPERS = 2

NOT_ENOUGH_PAPERS_MESSAGE = (
    "Not enough papers with abstracts were found. Try a broader topic, "
    "a lower minimum relevance score, or a wider year range in Settings."
)

# The stages that report progress, in the order they normally run.
STAGES = ["retriever", "synthesizer", "critic", "hypothesis", "novelty", "evaluator"]


class PipelineState(TypedDict, total=False):
    """Everything that travels between the agents."""
    # inputs
    topic: str
    settings: Dict[str, Any]
    force_fresh: bool
    run_discrimination: bool
    started_at: float
    # results of each agent
    papers: List[Dict[str, Any]]
    synthesis: Dict[str, Any]
    critic_result: Dict[str, Any]
    discrimination_result: Optional[Dict[str, Any]]
    hyp_result: Dict[str, Any]
    novelty_result: Dict[str, Any]
    # bookkeeping and output
    timings: Dict[str, float]
    error: Optional[str]
    final_report: Dict[str, Any]


# ---------------------------------------------------------------------------
# the real agents (imported lazily so this module can be tested without them)
# ---------------------------------------------------------------------------
def _count(n, singular, plural=None):
    """1 -> '1 theme', 4 -> '4 themes'."""
    return f"{n} {singular}" if n == 1 else f"{n} {plural or singular + 's'}"


def _default_agent(name):
    if name == "retrieve":
        from retriever import retrieve_papers
        return retrieve_papers
    if name == "synthesize":
        from synthesizer import synthesize
        return synthesize
    if name == "critic":
        from critic import run_critic_loop
        return run_critic_loop
    if name == "discrimination":
        from critic import run_discrimination_test
        return run_discrimination_test
    if name == "hypotheses":
        from hypothesis_generator import generate_hypotheses
        return generate_hypotheses
    if name == "novelty":
        from novelty_checker import check_novelty
        return check_novelty
    if name == "evaluate":
        return evaluator.evaluate
    raise KeyError(f"Unknown agent '{name}'")


def preload_agents():
    """
    Imports every real agent now. The web app calls this at start-up so the
    language models are loaded once, before the first search, instead of
    making the first search slow.
    """
    for name in ("retrieve", "synthesize", "critic", "hypotheses", "novelty", "evaluate"):
        _default_agent(name)


# ---------------------------------------------------------------------------
# report assembly
# ---------------------------------------------------------------------------
def build_report(state):
    """
    Assembles the final report from the shared state.

    The themes, gaps and contradictions come from the Critic's
    approved_synthesis - the version AFTER unsupported sentences were
    removed - not from the Synthesizer's raw output. That keeps the report
    consistent with the support ratio and the hypotheses (which are built
    from the approved synthesis), whether or not the Critic result was
    loaded from the cache.
    """
    critic_result = state["critic_result"]
    synthesis = state["synthesis"]
    approved = critic_result.get("approved_synthesis") or synthesis
    themes = approved.get("themes", synthesis.get("themes", {}))

    return {
        "topic": state["topic"],
        "settings": state["settings"],
        "num_papers": len(state["papers"]),
        "papers": state["papers"],
        "num_themes": len(themes),
        "themes": themes,
        "gaps": approved.get("gaps", synthesis.get("gaps", [])),
        "contradictions": approved.get("contradictions", synthesis.get("contradictions", [])),
        "critic_approved": critic_result["approved"],
        "critic_iterations": critic_result["iterations_used"],
        "critic_support_ratio": critic_result["final_verdict"]["support_ratio"],
        "discrimination_test": state.get("discrimination_result"),
        "hypotheses": state["hyp_result"]["hypotheses"],
        "key_terms": state["hyp_result"].get("key_terms_extracted", []),
        "novelty_check": state["novelty_result"]["per_hypothesis"],
    }


# ---------------------------------------------------------------------------
# the graph
# ---------------------------------------------------------------------------
def build_graph(progress: Optional[Callable[[str, str, str], None]] = None,
                agents: Optional[Dict[str, Callable]] = None):
    """
    Builds and compiles the pipeline graph.

    progress : optional function(stage, status, message) for live updates.
    agents   : optional dict that replaces any real agent by name
               (retrieve, synthesize, critic, discrimination, hypotheses,
               novelty, evaluate). Used by the tests; normally left empty.
    """
    overrides = agents or {}

    def agent(name):
        return overrides[name] if name in overrides else _default_agent(name)

    def emit(stage, status, message=""):
        if progress:
            progress(stage, status, message)

    def add_timing(state, stage, started):
        timings = dict(state.get("timings") or {})
        timings[stage] = round(time.time() - started, 2)
        return timings

    # ----- nodes ---------------------------------------------------------
    def retriever_node(state):
        emit("retriever", "running", "Fetching papers from OpenAlex...")
        started = time.time()
        try:
            papers = agent("retrieve")(state["topic"], settings=state["settings"],
                                       force_refresh=state["force_fresh"])
        except Exception as e:
            emit("retriever", "error", str(e))
            raise
        timings = add_timing(state, "retriever", started)

        if len(papers) < MIN_PAPERS:
            emit("retriever", "error", "Not enough papers found for this topic/settings.")
            return {"papers": papers, "timings": timings, "error": NOT_ENOUGH_PAPERS_MESSAGE}

        emit("retriever", "done", f"{_count(len(papers), 'relevant paper')} retrieved")
        return {"papers": papers, "timings": timings}

    def synthesizer_node(state):
        emit("synthesizer", "running", "Clustering papers into themes...")
        started = time.time()
        try:
            synthesis = agent("synthesize")(state["topic"], settings=state["settings"],
                                            force_refresh=state["force_fresh"])
        except Exception as e:
            emit("synthesizer", "error", str(e))
            raise
        emit("synthesizer", "done", f"{_count(len(synthesis['themes']), 'theme')} identified")
        return {"synthesis": synthesis, "timings": add_timing(state, "synthesizer", started)}

    def critic_node(state):
        emit("critic", "running", "Verifying claims against source abstracts...")
        started = time.time()
        try:
            result = agent("critic")(state["synthesis"], state["topic"],
                                     settings=state["settings"], force_refresh=state["force_fresh"])
        except Exception as e:
            emit("critic", "error", str(e))
            raise
        verdict = "Approved" if result["approved"] else "Best effort (max iterations)"
        emit("critic", "done",
             f"{verdict} \u2014 {result['iterations_used']} iteration(s), "
             f"support ratio {result['final_verdict']['support_ratio']}")
        return {"critic_result": result, "timings": add_timing(state, "critic", started)}

    def discrimination_node(state):
        emit("discrimination", "running", "Running the genuine-vs-corrupted claim test...")
        started = time.time()
        try:
            result = agent("discrimination")(state["topic"], settings=state["settings"],
                                             force_refresh=state["force_fresh"])
        except Exception as e:
            emit("discrimination", "error", str(e))
            raise
        emit("discrimination", "done", f"Accuracy {result.get('accuracy')}")
        return {"discrimination_result": result,
                "timings": add_timing(state, "critic_discrimination_test", started)}

    def hypothesis_node(state):
        emit("hypothesis", "running", "Extracting key terms and drafting hypotheses...")
        started = time.time()
        try:
            result = agent("hypotheses")(state["critic_result"], settings=state["settings"],
                                         force_refresh=state["force_fresh"])
        except Exception as e:
            emit("hypothesis", "error", str(e))
            raise
        emit("hypothesis", "done", f"{_count(len(result['hypotheses']), 'hypothesis', 'hypotheses')} generated")
        return {"hyp_result": result, "timings": add_timing(state, "hypothesis_generator", started)}

    def novelty_node(state):
        emit("novelty", "running", "Checking hypotheses against published work...")
        started = time.time()
        try:
            result = agent("novelty")(state["hyp_result"], settings=state["settings"],
                                      force_refresh=state["force_fresh"])
        except Exception as e:
            emit("novelty", "error", str(e))
            raise
        emit("novelty", "done",
             f"{result['num_hypotheses_checked']} checked, "
             f"{result['num_close_matches_found']} close match(es) found")
        return {"novelty_result": result, "timings": add_timing(state, "novelty_checker", started)}

    def evaluator_node(state):
        emit("evaluator", "running", "Scoring coverage and hypothesis specificity...")
        started = time.time()
        report = build_report(state)

        # Scoring is a bonus: if it ever fails, the finished run must not be lost.
        try:
            evaluation = agent("evaluate")(report)
            overall = evaluation["overall"]
            message = (f"Overall {overall['score']} ({overall['label']})"
                       if overall["score"] is not None else "Nothing to score")
            emit("evaluator", "done", message)
        except Exception as e:
            import traceback
            traceback.print_exc()
            evaluation = None
            emit("evaluator", "done", "Scoring unavailable for this run (see the terminal for details)")
        report["evaluation"] = evaluation

        timings = add_timing(state, "evaluator", started)
        timings["total"] = round(time.time() - state.get("started_at", started), 2)
        report["timings_seconds"] = timings
        return {"final_report": report, "timings": timings}

    # ----- routing decisions ----------------------------------------------
    def after_retriever(state):
        return "stop" if state.get("error") else "continue"

    def after_critic(state):
        return "discrimination" if state.get("run_discrimination") else "hypothesis"

    # ----- wiring -----------------------------------------------------------
    graph = StateGraph(PipelineState)
    graph.add_node("retriever", retriever_node)
    graph.add_node("synthesizer", synthesizer_node)
    graph.add_node("critic", critic_node)
    graph.add_node("discrimination", discrimination_node)
    graph.add_node("hypothesis", hypothesis_node)
    graph.add_node("novelty", novelty_node)
    graph.add_node("evaluator", evaluator_node)

    graph.add_edge(START, "retriever")
    graph.add_conditional_edges("retriever", after_retriever,
                                {"continue": "synthesizer", "stop": END})
    graph.add_edge("synthesizer", "critic")
    graph.add_conditional_edges("critic", after_critic,
                                {"discrimination": "discrimination", "hypothesis": "hypothesis"})
    graph.add_edge("discrimination", "hypothesis")
    graph.add_edge("hypothesis", "novelty")
    graph.add_edge("novelty", "evaluator")
    graph.add_edge("evaluator", END)

    return graph.compile()


def run_graph(topic, settings=None, force_fresh=False, run_discrimination=False,
              progress=None, agents=None):
    """
    Runs the whole pipeline on a topic and returns the final state.

    On success the state contains "final_report". If the run stopped early
    (too few papers) it contains "error" and no "final_report". Any other
    failure inside an agent is raised as an exception, after a progress
    "error" event for that stage.
    """
    compiled = build_graph(progress=progress, agents=agents)
    initial_state: PipelineState = {
        "topic": topic,
        "settings": config.merge_settings(settings),
        "force_fresh": bool(force_fresh),
        "run_discrimination": bool(run_discrimination),
        "started_at": time.time(),
        "timings": {},
        "error": None,
    }
    return compiled.invoke(initial_state)


def graph_as_mermaid():
    """The pipeline drawn as Mermaid text - paste it into any Mermaid viewer
    (for example mermaid.live) to get a diagram for the project report."""
    return build_graph(agents={}).get_graph().draw_mermaid()


if __name__ == "__main__":
    print(graph_as_mermaid())