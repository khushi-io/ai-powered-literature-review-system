"""
test_evaluator.py
Self-test for evaluator.py.

PART 1 - unit checks on small made-up reports where the correct score
         is known in advance (fast, no model needed).
PART 2 - a real demo: scores your actual cached pipeline results and
         prints each score together with the reason for it.

Run with:  python test_evaluator.py
Nothing is written to your real data (the history check uses a
temporary folder).
"""

import os
import copy
import json
import tempfile

import config
import evaluator
import history

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
# tiny made-up literature about "network defense"
# ---------------------------------------------------------------------------
S1 = "Firewalls inspect packets and block suspicious connections before they reach internal servers."
S2 = "Intrusion detection systems raise alerts when traffic patterns match known malware signatures."
S3 = "Encryption protects stored credentials and prevents attackers from reading intercepted passwords."

PAPERS = [
    {"title": "Firewall design", "abstract": f"Abstract: {S1} Rules are updated weekly by administrators."},
    {"title": "Detecting intrusions", "abstract": f"{S2} Several detectors were compared on benchmark traffic."},
    {"title": "Password safety", "abstract": f"{S3} Users often reuse the same password everywhere."},
]


def theme_with(*sentences):
    return {"paper_titles": ["x"], "paper_count": len(sentences), "summary_sentences": list(sentences)}


def make_report(**overrides):
    report = {
        "topic": "network defense",
        "papers": copy.deepcopy(PAPERS),
        "themes": {"0": theme_with(S1, S2)},                  # papers 1 and 2 represented, paper 3 not
        "hypotheses": [],
        "critic_approved": True,
        "critic_support_ratio": 0.9,
        "novelty_check": [],
    }
    report.update(overrides)
    return report


def hyp(hid, text, grounding):
    return {"id": hid, "hypothesis": text, "grounding_terms": grounding}


GOOD_HYP = hyp("H1", "The effect of firewalls on intrusion in the context of network defense remains "
                     "insufficiently studied; it is hypothesized that firewalls reduce intrusion.",
               ["firewalls", "intrusion"])

# ===========================================================================
print("PART 1 - unit checks\n")

print("1. Settings are sane")
check("coverage weights add up to 1", abs(sum(evaluator.COVERAGE_WEIGHTS.values()) - 1) < 1e-9)
check("specificity weights add up to 1", abs(sum(evaluator.SPECIFICITY_WEIGHTS.values()) - 1) < 1e-9)

print("\n2. Coverage - which papers the summaries really use")
cov = evaluator.evaluate(make_report())["coverage"]
check("2 of 3 papers are represented", cov["papers_represented"] == 2 and cov["papers_total"] == 3)
check("paper_representation = 0.667", cov["paper_representation"] == 0.667)
all_three = evaluator.evaluate(make_report(themes={"0": theme_with(S1, S2), "1": theme_with(S3)}))["coverage"]
check("all 3 papers represented when every paper contributes", all_three["paper_representation"] == 1.0)
edited = evaluator.evaluate(make_report(themes={"0": theme_with("Firewalls inspect packets, and block suspicious connections before they reach internal servers")}))["coverage"]
check("a lightly edited sentence (punctuation changed) still counts", edited["papers_represented"] == 1)
invented = evaluator.evaluate(make_report(themes={"0": theme_with("Quantum computers will make every existing cipher obsolete within a decade.")}))["coverage"]
check("an invented sentence matches no paper", invented["papers_represented"] == 0 and invented["paper_representation"] == 0.0)

print("\n3. Coverage - characteristic terms")
terms = evaluator.corpus_key_terms(PAPERS, "network defense")
check("key terms are found", len(terms) >= 5)
check("topic words are excluded (network, defense)", "network" not in terms and "defense" not in terms)
check("subject words are included", any(t in terms for t in ("firewalls", "intrusion", "encryption", "malware", "passwords")), terms)
ethics_terms = evaluator.corpus_key_terms(
    [{"title": "Hackers and ethics", "abstract": "Ethical hackers practise ethics. Hacking tools find vulnerabilities in networks."}] * 2, "Ethical hacking")
check("variants of topic words are excluded (hackers, ethics for 'Ethical hacking')", "hackers" not in ethics_terms and "ethics" not in ethics_terms, ethics_terms)
ai_terms = evaluator.corpus_key_terms(
    [{"title": "Artificial intelligence", "abstract": "Artificial intelligence raises accountability questions for regulators."}] * 2, "AI regulation")
check("acronym expansion is excluded ('AI' -> artificial, intelligence)", "artificial" not in ai_terms and "intelligence" not in ai_terms, ai_terms)
generic_terms = evaluator.corpus_key_terms(
    [{"title": "x", "abstract": "This good paper presents the best results. Firewalls matter."}] * 2, "security")
check("generic words (good, best, paper, results) are excluded", not ({"good", "best", "paper", "results"} & set(generic_terms)), generic_terms)
full = evaluator.evaluate(make_report(themes={"0": theme_with(S1, S2, S3)}))["coverage"]
check("term_coverage is between 0 and 1 and lists covered/missing terms",
      0 <= full["term_coverage"] <= 1 and len(full["terms_covered"]) + len(full["terms_missing"]) == len(full["top_terms"]))
none_cov = evaluator.evaluate(make_report(themes={"0": theme_with("Quantum computers will make every existing cipher obsolete within a decade.")}))["coverage"]
check("a summary that mentions none of the papers' terms scores low", none_cov["term_coverage"] <= 0.2, none_cov["term_coverage"])

print("\n4. Specificity - one check at a time")
def spec_of(h, **kw):
    return evaluator.evaluate(make_report(hypotheses=[h], **kw))["specificity"]["per_hypothesis"][0]

good = spec_of(GOOD_HYP)
check("a concrete, testable hypothesis scores high (>= 0.9)", good["score"] >= 0.9, good)
check("it needs no warnings", good["notes"] == [])

generic = spec_of(hyp("H1", "The effect of big on good in the context of network defense remains insufficiently studied.", ["big", "good"]))
check("generic grounding words score 0 on grounding_quality", generic["checks"]["grounding_quality"] == 0.0)
check("and the notes name them", any("big" in n and "good" in n for n in generic["notes"]), generic["notes"])
check("a generic hypothesis scores clearly lower than a concrete one", generic["score"] < good["score"] - 0.3)

number = spec_of(hyp("H1", "The effect of 42 on firewalls in network defense remains open.", ["42", "firewalls"]))
check("a bare number is not a substantive grounding term (0.5)", number["checks"]["grounding_quality"] == 0.5)

invented_term = spec_of(hyp("H1", "The effect of quantum on firewalls in network defense remains open.", ["quantum", "firewalls"]))
check("a grounding term absent from the papers lowers corpus_grounded (0.5)", invented_term["checks"]["corpus_grounded"] == 0.5)
check("and is named in the notes", any("quantum" in n for n in invented_term["notes"]))

no_relation = spec_of(hyp("H1", "Firewalls and intrusion are both subjects in network defense.", ["firewalls", "intrusion"]))
check("no relationship wording -> testable_relation 0", no_relation["checks"]["testable_relation"] == 0.0)

one_variable = spec_of(hyp("H1", "The effect of firewalls in network defense remains open.", ["firewalls", "intrusion"]))
check("relationship but only one grounding term named -> 0.5", one_variable["checks"]["testable_relation"] == 0.5)

off_topic = spec_of(hyp("H1", "The effect of firewalls on intrusion remains open to question.", ["firewalls", "intrusion"]))
check("a hypothesis that never mentions the topic -> on_topic 0", off_topic["checks"]["on_topic"] == 0.0)

no_grounding = spec_of(hyp("H1", "The effect of something on network defense remains open.", []))
check("no key terms -> grounding_quality 0 and a note", no_grounding["checks"]["grounding_quality"] == 0.0 and "no key terms" in no_grounding["notes"])

print("\n5. Specificity - duplicates")
twin_a = hyp("H1", "The effect of firewalls on intrusion in network defense remains open.", ["firewalls", "intrusion"])
twin_b = hyp("H2", "The effect of intrusion on firewalls in network defense remains open.", ["intrusion", "firewalls"])
twins = evaluator.evaluate(make_report(hypotheses=[twin_a, twin_b]))["specificity"]
check("two hypotheses about the same variables have distinct = 0", all(p["checks"]["distinct"] == 0.0 for p in twins["per_hypothesis"]))
check("a weakness mentions the similar hypotheses", any("similar" in w.lower() for w in evaluator.evaluate(make_report(hypotheses=[twin_a, twin_b]))["weaknesses"]))
different = hyp("H2", "The effect of encryption on passwords in network defense remains open.", ["encryption", "passwords"])
mixed = evaluator.evaluate(make_report(hypotheses=[twin_a, different]))["specificity"]
check("hypotheses about different variables are fully distinct", all(p["checks"]["distinct"] == 1.0 for p in mixed["per_hypothesis"]))
check("specificity score is the average of the per-hypothesis scores",
      abs(mixed["score"] - sum(p["score"] for p in mixed["per_hypothesis"]) / 2) < 0.002)
check("generic words are collected for the report", "big" in evaluator.evaluate(make_report(hypotheses=[hyp("H1", "x big y network defense effect of big on firewalls", ["big", "firewalls"])]))["specificity"]["flagged_terms"])

print("\n6. Support, novelty and overall")
r = evaluator.evaluate(make_report(hypotheses=[GOOD_HYP],
                                   novelty_check=[{"hypothesis_id": "H1", "checked": True, "is_close_match": True}]))
check("support = the Critic's ratio", r["support"]["score"] == 0.9)
check("novelty = share without a close match (0 of 1 -> 0.0)", r["novelty"]["score"] == 0.0 and r["novelty"]["close_matches"] == 1)
two = evaluator.evaluate(make_report(hypotheses=[GOOD_HYP, different],
                                     novelty_check=[{"checked": True, "is_close_match": False}, {"checked": True, "is_close_match": True}]))
check("novelty with 1 of 2 overlapping -> 0.5", two["novelty"]["score"] == 0.5)
parts = [two["coverage"]["score"], two["specificity"]["score"], two["support"]["score"], two["novelty"]["score"]]
check("overall = plain average of the four scores", abs(two["overall"]["score"] - sum(parts) / 4) < 0.002)
check("overall lists which scores it used", two["overall"]["components_used"] == ["coverage", "novelty", "specificity", "support"])
check("a possible-overlap weakness is reported", any("overlap" in w for w in r["weaknesses"]))

print("\n7. Labels")
for value, expected in [(0.75, "strong"), (0.7499, "fair"), (0.5, "fair"), (0.4999, "weak"), (0.0, "weak"), (1.0, "strong"), (None, None)]:
    check(f"label({value}) = {expected}", evaluator._label(value) == expected)

print("\n8. Missing and empty data never crash")
empty = evaluator.evaluate({})
check("an empty report gives no scores and no overall", empty["overall"]["score"] is None and empty["coverage"] is None
      and empty["specificity"] is None and empty["support"] is None and empty["novelty"] is None)
check("an empty report has no weaknesses", empty["weaknesses"] == [])
only_hyp = evaluator.evaluate({"topic": "network defense", "hypotheses": [GOOD_HYP]})
check("only hypotheses: overall = specificity alone", only_hyp["overall"]["components_used"] == ["specificity"]
      and only_hyp["overall"]["score"] == only_hyp["specificity"]["score"])
only_papers = evaluator.evaluate({"topic": "network defense", "papers": copy.deepcopy(PAPERS)})
check("papers but no themes: coverage is 0, not a crash", only_papers["coverage"]["paper_representation"] == 0.0)
no_abstracts = evaluator.evaluate(make_report(papers=[{"title": "A"}, {"title": "B"}]))
check("papers without abstracts do not crash", no_abstracts["coverage"]["papers_represented"] == 0)
check("support ratio missing -> support is None", evaluator.evaluate(make_report(critic_support_ratio=None))["support"] is None)
check("themes with int keys also work", evaluator.evaluate(make_report(themes={0: theme_with(S1)}))["coverage"]["papers_represented"] == 1)

print("\n9. Output hygiene")
report = make_report(hypotheses=[GOOD_HYP])
before = copy.deepcopy(report)
result = evaluator.evaluate(report)
check("result is JSON-serialisable", bool(json.dumps(result)))
check("the input report is never modified", report == before)
check("every score is between 0 and 1", all(0 <= x <= 1 for x in [
    result["coverage"]["score"], result["specificity"]["score"], result["support"]["score"], result["overall"]["score"]]))
check("'how_scored' explains 6 topics in plain words", len(result["how_scored"]) == 6 and all(i["term"] and i["text"] for i in result["how_scored"]))
check("weaknesses are plain sentences", all(isinstance(w, str) for w in result["weaknesses"]))
check("same input gives the same output (deterministic)", evaluator.evaluate(report) == result)

print("\n10. Works on a report loaded back from the Search History")
config.HISTORY_DIR = tempfile.mkdtemp(prefix="eval_history_")
saved_report = make_report(hypotheses=[GOOD_HYP])
saved_report["settings"] = config.merge_settings(None)
run_id = history.save_run(saved_report)
loaded = history.load_run(run_id)["report"]
check("scores from the saved copy equal scores from the original", evaluator.evaluate(loaded) == evaluator.evaluate(saved_report))

# ===========================================================================
print("\n" + "=" * 70)
print("PART 2 - real demo on your cached results")
print("=" * 70)


def load_json(*parts):
    path = os.path.join(config.CACHE_DIR, *parts)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def report_from_cache(slug):
    papers = load_json("retriever", f"{slug}.json")
    synth = load_json("synthesizer", f"{slug}.json")
    critic = load_json("critic", f"{slug}.json")
    hyp_ = load_json("hypothesis", f"{slug}.json")
    if not (papers and synth and critic and hyp_):
        return None
    novelty = load_json("novelty", f"{slug}.json") or {}
    return {
        "topic": synth["topic"], "num_papers": len(papers), "papers": papers,
        "themes": synth["themes"], "gaps": synth.get("gaps", []),
        "critic_approved": critic["approved"],
        "critic_support_ratio": critic["final_verdict"]["support_ratio"],
        "hypotheses": hyp_["hypotheses"], "key_terms": hyp_["key_terms_extracted"],
        "novelty_check": novelty.get("per_hypothesis", []),
    }


demo_ran = 0
for slug in ["ethical_ai", "ethical_hacking", "unethical_ai", "machine_learning_for_crop_yield_prediction"]:
    rep = report_from_cache(slug)
    if not rep:
        print(f"\n(skipping {slug}: cached results not found)")
        continue
    ev = evaluator.evaluate(rep)
    demo_ran += 1
    print(f"\n--- {rep['topic']} ---")
    print(f"  OVERALL {ev['overall']['score']} ({ev['overall']['label']})  from {', '.join(ev['overall']['components_used'])}")
    c = ev["coverage"]
    print(f"  coverage {c['score']} ({c['label']}): {c['papers_represented']} of {c['papers_total']} papers used, "
          f"{len(c['terms_covered'])} of {len(c['top_terms'])} key terms covered")
    print(f"     key terms : {', '.join(c['top_terms'])}")
    print(f"     not covered: {', '.join(c['terms_missing']) or 'none'}")
    s = ev["specificity"]
    print(f"  specificity {s['score']} ({s['label']})")
    for p in s["per_hypothesis"]:
        print(f"     {p['id']}: {p['score']} ({p['label']})  {'; '.join(p['notes']) or 'no warnings'}")
    print(f"  support {ev['support']['score'] if ev['support'] else 'n/a'} | "
          f"novelty {ev['novelty']['score'] if ev['novelty'] else 'not checked'}")
    for w in ev["weaknesses"]:
        print("  ! " + w)
    check(f"{slug}: scores are valid and JSON-serialisable", bool(json.dumps(ev)) and 0 <= ev["overall"]["score"] <= 1)

check("at least one real report was scored", demo_ran >= 1)

print("\n" + "=" * 70)
print(f"RESULT: {checks_passed} passed, {checks_failed} failed")
if checks_failed == 0:
    print("All evaluator.py checks passed.")