"""
test_comparison.py
Self-test for comparison.py.

PART 1 - unit checks with a tiny fake embedding function (fast, no
         model needed, results are fully predictable).
PART 2 - a real demo: compares your actual cached pipeline results
         (e.g. Ethical AI vs Ethical hacking) using the REAL embedding
         model, and prints what the comparison finds. This part loads
         the model, so it takes a few seconds.

Run with:  python test_comparison.py
Nothing is written to disk.
"""

import os
import re
import copy
import json
import zlib

import numpy as np

import config
import comparison

checks_passed = 0
checks_failed = 0


def check(name, condition):
    global checks_passed, checks_failed
    if condition:
        checks_passed += 1
        print(f"  PASS  {name}")
    else:
        checks_failed += 1
        print(f"  FAIL  {name}")


def fake_embed(texts):
    """Bag-of-words hashed into 128 dimensions: texts sharing words get similar vectors."""
    out = np.zeros((len(texts), 128))
    for i, text in enumerate(texts):
        for word in re.findall(r"[a-z]+", text.lower()):
            if len(word) > 3:
                out[i, zlib.crc32(word.encode()) % 128] += 1
    return out


def make_report(topic, papers, themes, key_terms=None, hypotheses=None,
                approved=True, novelty=None):
    return {
        "topic": topic,
        "num_papers": len(papers),
        "papers": papers,
        "themes": themes,
        "gaps": [], "contradictions": [],
        "critic_approved": approved, "critic_iterations": 1, "critic_support_ratio": 0.9,
        "hypotheses": hypotheses or [],
        "key_terms": key_terms or [],
        "novelty_check": novelty or [],
    }


def theme(sentence_text, n=3):
    return {"paper_titles": ["x"] * n, "paper_count": n, "summary_sentences": [sentence_text]}


# ===========================================================================
print("PART 1 - unit checks (fake embedding function)\n")

print("1. Shared papers")
papers_a = [
    {"title": "Alpha Study", "year": 2020, "doi": "https://doi.org/10.1/AAA", "cited_by_count": 10, "is_oa": True},
    {"title": "Beta: A Study!", "year": 2021, "doi": None, "cited_by_count": 20, "is_oa": False},
    {"title": "Gamma Work", "year": 2022, "doi": "https://doi.org/10.1/ccc", "cited_by_count": 30, "is_oa": True},
    {"title": "Alpha Study", "year": 2020, "doi": "https://doi.org/10.1/AAA"},          # duplicate inside A
]
papers_b = [
    {"title": "ALPHA STUDY (different title text)", "year": 2020, "doi": "10.1/aaa"},  # same DOI, other format
    {"title": "beta a study", "year": 2021, "doi": ""},                                  # same title, no DOI
    {"title": "Delta Only", "year": 2023, "doi": "https://doi.org/10.1/ddd"},
]
ra = make_report("topic A", papers_a, {"0": theme("alpha beta gamma testing")})
rb = make_report("topic B", papers_b, {"0": theme("delta testing alpha")})
res = comparison.compare_reports(ra, rb, embed_fn=fake_embed)
p = res["papers"]
check("shared by DOI (different formatting) and by title (no DOI) = 2", p["shared_count"] == 2)
check("duplicate paper inside A is counted once (union = 3 + 3 - 2 = 4)", p["union_count"] == 4)
check("Jaccard = 0.5", p["jaccard"] == 0.5)
check("shared list has titles, years and links",
      {s["title"] for s in p["shared"]} == {"Alpha Study", "Beta: A Study!"} and p["shared"][0]["year"] in (2020, 2021))
check("percentages are of each side's unique papers (A has 3, B has 3 -> 67%)", p["pct_of_a"] == 67 and p["pct_of_b"] == 67)
check("per-side unique paper counts are reported (3 and 3)", p["count_a"] == 3 and p["count_b"] == 3)

res0 = comparison.compare_reports(
    make_report("x", [{"title": "One", "doi": "10.1/x"}], {}),
    make_report("y", [{"title": "Two", "doi": "10.1/y"}], {}), embed_fn=fake_embed)
check("no false matches between unrelated papers", res0["papers"]["shared_count"] == 0 and res0["papers"]["jaccard"] == 0.0)

print("\n2. Headline statistics")
a = res["a"]
check("year range from the papers", a["year_min"] == 2020 and a["year_max"] == 2022)
check("average citations (10, 20, 30 and one paper without a value)", a["avg_citations"] == 20.0)
check("open-access share (2 of 3 papers that have the field = 67%)", a["open_access_pct"] == 67)
check("open-access share is None (not 0) when data predates the field", res["b"]["open_access_pct"] is None)

print("\n3. Key terms")
r1 = make_report("a", [], {}, key_terms=["IoT", "bias", "Fairness"])
r2 = make_report("b", [], {}, key_terms=["iot", "privacy"])
kt = comparison.compare_reports(r1, r2, embed_fn=fake_embed)["key_terms"]
check("shared terms are matched ignoring case", kt["shared"] == ["IoT"])
check("unique terms per side", kt["only_a"] == ["bias", "Fairness"] and kt["only_b"] == ["privacy"])
r3 = make_report("a", [], {}, hypotheses=[{"grounding_terms": ["chatgpt", "iot"]}, {"grounding_terms": ["iot", "bard"]}])
kt3 = comparison.compare_reports(r3, r2, embed_fn=fake_embed)["key_terms"]
check("falls back to hypothesis grounding terms when key_terms is missing, without duplicates",
      kt3["shared"] == ["iot"] and kt3["only_a"] == ["chatgpt", "bard"])

print("\n4. Theme alignment")
themes_a = {"0": theme("ethical hackers perform penetration testing to find network security attacks"),
            "1": theme("farmers predict crop yield using soil moisture and weather satellite data")}
themes_b = {"0": theme("penetration testing finds network security attacks and vulnerabilities"),
            "1": theme("medical imaging segmentation of tumor regions with convolutional networks")}
th = comparison.compare_reports(make_report("a", [], themes_a), make_report("b", [], themes_b), embed_fn=fake_embed)["themes"]
al = {x["a_theme"]: x for x in th["alignments"]}
check("matching themes are 'aligned' and point at each other", al["0"]["relation"] == "aligned" and al["0"]["b_theme"] == "0")
check("the best match is mutual", al["0"]["mutual"] is True)
check("unrelated theme is 'distinct'", al["1"]["relation"] == "distinct")
check("unique-to-A lists the farming theme", [x["a_theme"] for x in th["unique_a"]] == ["1"])
check("unique-to-B lists the medical-imaging theme", [x["b_theme"] for x in th["unique_b"]] == ["1"])
check("similarity matrix is 2 x 2 with values between -1 and 1",
      len(th["matrix"]) == 2 and len(th["matrix"][0]) == 2 and all(-1.0001 <= v <= 1.0001 for row in th["matrix"] for v in row))
check("method is reported", th["method"] == "custom embedding function")
check("theme paper counts are reported per side", th["a_papers"] == [3, 3] and th["b_papers"] == [3, 3])
check("summary spells out aligned / related / unique counts",
      any("aligned" in line and "related" in line and "unique" in line
          for line in comparison.compare_reports(make_report("a", [], themes_a), make_report("b", [], themes_b), embed_fn=fake_embed)["summary"]))

print("\n5. Comparing a search with itself")
same = comparison.compare_reports(ra, copy.deepcopy(ra), embed_fn=fake_embed)
check("all papers shared, Jaccard 1.0", same["papers"]["jaccard"] == 1.0)
check("every theme is aligned with similarity ~1", all(x["relation"] == "aligned" and x["similarity"] > 0.99 for x in same["themes"]["alignments"]))

print("\n6. Edge cases")
empty = comparison.compare_reports(make_report("e1", [], {}), make_report("e2", [], {}), embed_fn=fake_embed)
check("two empty reports do not crash", empty["papers"]["shared_count"] == 0 and empty["themes"]["alignments"] == [])
one_sided = comparison.compare_reports(make_report("e1", [], {}), make_report("e2", [], themes_b), embed_fn=fake_embed)
check("a search with no themes still compares", one_sided["themes"]["unique_b"] != [] and len(one_sided["summary"]) >= 3)
int_keys = comparison.compare_reports(make_report("a", [], {0: theme("alpha testing words here")}),
                                      make_report("b", [], {0: theme("alpha testing words here")}), embed_fn=fake_embed)
check("theme ids may be ints (in-memory) as well as strings (loaded from history)",
      int_keys["themes"]["alignments"][0]["relation"] == "aligned")
check("themes with only paper titles (no sentences) still get compared",
      comparison.compare_reports(make_report("a", [], {"0": {"paper_titles": ["Deep nets for images"], "paper_count": 1, "summary_sentences": []}}),
                                 make_report("b", [], themes_b), embed_fn=fake_embed)["themes"]["alignments"][0]["b_theme"] is not None)

print("\n7. Fallback when the embedding model fails")
def broken_embed(texts):
    raise RuntimeError("model unavailable (simulated)")
fb = comparison.compare_reports(make_report("a", [], themes_a), make_report("b", [], themes_b), embed_fn=broken_embed)
check("falls back to TF-IDF instead of crashing", "TF-IDF" in fb["themes"]["method"])
check("fallback still finds the matching themes", {x["a_theme"]: x for x in fb["themes"]["alignments"]}["0"]["b_theme"] == "0")
def wrong_shape_embed(texts):
    return np.ones((1, 8))
check("an embedder that returns the wrong number of vectors also falls back",
      "TF-IDF" in comparison.compare_reports(make_report("a", [], themes_a), make_report("b", [], themes_b), embed_fn=wrong_shape_embed)["themes"]["method"])

print("\n7b. Quality scores from the Evaluator")
def with_eval(report, overall, label="strong", cov=0.8, spec=0.9, sup=1.0, nov=0.5):
    report = copy.deepcopy(report)
    report["evaluation"] = {"overall": {"score": overall, "label": label},
                            "coverage": {"score": cov}, "specificity": {"score": spec},
                            "support": {"score": sup}, "novelty": {"score": nov}}
    return report
qa = comparison.compare_reports(with_eval(make_report("a", [], themes_a), 0.9), with_eval(make_report("b", [], themes_b), 0.7), embed_fn=fake_embed)
check("evaluation scores appear in each side's headline numbers",
      qa["a"]["evaluation_overall"] == 0.9 and qa["b"]["evaluation_overall"] == 0.7
      and qa["a"]["evaluation_coverage"] == 0.8 and qa["a"]["evaluation_specificity"] == 0.9
      and qa["a"]["evaluation_support"] == 1.0 and qa["a"]["evaluation_novelty"] == 0.5)
check("summary says which search scored higher", any("Search A has the higher overall quality score (0.9 against 0.7)" in l for l in qa["summary"]))
qb = comparison.compare_reports(with_eval(make_report("a", [], themes_a), 0.6), with_eval(make_report("b", [], themes_b), 0.8), embed_fn=fake_embed)
check("it names B when B scored higher", any("Search B has the higher" in l for l in qb["summary"]))
qs = comparison.compare_reports(with_eval(make_report("a", [], themes_a), 0.75), with_eval(make_report("b", [], themes_b), 0.75), embed_fn=fake_embed)
check("equal scores are described as equal", any("same overall quality score (0.75)" in l for l in qs["summary"]))
qn = comparison.compare_reports(make_report("a", [], themes_a), with_eval(make_report("b", [], themes_b), 0.8), embed_fn=fake_embed)
check("a run without an evaluation gives None and no quality sentence",
      qn["a"]["evaluation_overall"] is None and not any("quality score" in l for l in qn["summary"]))

print("\n8. Output hygiene")
check("result is JSON-serialisable", bool(json.dumps(res)))
before_a, before_b = copy.deepcopy(ra), copy.deepcopy(rb)
comparison.compare_reports(ra, rb, embed_fn=fake_embed)
check("input reports are never modified", ra == before_a and rb == before_b)
check("summary is plain sentences mentioning both topics",
      all(isinstance(s, str) for s in res["summary"]) and "topic A" in res["summary"][0] and "topic B" in res["summary"][0])
check("method note says scores are a guide, not proof", "not proof" in res["method_note"])
htr = res["how_to_read"]
check("'how to read' has 7 labelled explanations", len(htr) == 7 and all(item["term"] and item["text"] for item in htr))
check("'how to read' explains the quality scores", any(i["term"] == "Quality scores" for i in htr))
check("'how to read' explains DOIs in plain words", any("DOI" in i["text"] and "permanent ID" in i["text"] for i in htr))
check("'how to read' quotes the real thresholds", any("0.55" in i["text"] and "0.35" in i["text"] for i in htr))
fb_htr = comparison.compare_reports(make_report("a", [], themes_a), make_report("b", [], themes_b), embed_fn=broken_embed)["how_to_read"]
check("explanation changes when the TF-IDF fallback is used", any("TF-IDF" in i["text"] for i in fb_htr))
one = comparison.compare_reports(make_report("a", [{"title": "Solo"}], {"0": theme("alpha testing words here")}),
                                 make_report("b", [{"title": "Solo"}], {"0": theme("alpha testing words here")}), embed_fn=fake_embed)
check("singular wording: '1 paper in 1 theme' (not '1 papers')",
      "1 paper in 1 theme" in one["summary"][0] and "1 papers" not in " ".join(one["summary"]) and "1 themes" not in " ".join(one["summary"]))

# ===========================================================================
print("\n" + "=" * 70)
print("PART 2 - real demo on your cached results (uses the real embedding model)")
print("=" * 70)


def load_json(*parts):
    path = os.path.join(config.CACHE_DIR, *parts)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def report_from_cache(slug):
    """Rebuilds a report like app.py produces, from the stage caches of a default-settings run."""
    papers = load_json("retriever", f"{slug}.json")
    synth = load_json("synthesizer", f"{slug}.json")
    critic = load_json("critic", f"{slug}.json")
    hyp = load_json("hypothesis", f"{slug}.json")
    if not (papers and synth and critic and hyp):
        return None
    novelty = load_json("novelty", f"{slug}.json") or {}
    return {
        "topic": synth["topic"],
        "num_papers": len(papers), "papers": papers,
        "themes": synth["themes"], "gaps": synth.get("gaps", []),
        "contradictions": synth.get("contradictions", []),
        "critic_approved": critic["approved"], "critic_iterations": critic["iterations_used"],
        "critic_support_ratio": critic["final_verdict"]["support_ratio"],
        "hypotheses": hyp["hypotheses"], "key_terms": hyp["key_terms_extracted"],
        "novelty_check": novelty.get("per_hypothesis", []),
    }


real_pairs = [("ethical_ai", "ethical_hacking"), ("ethical_ai", "unethical_ai"),
              ("machine_learning_for_crop_yield_prediction", "ethical_hacking")]
demo_ran = 0
for slug_a, slug_b in real_pairs:
    rep_a, rep_b = report_from_cache(slug_a), report_from_cache(slug_b)
    if not rep_a or not rep_b:
        print(f"\n(skipping {slug_a} vs {slug_b}: cached results not found)")
        continue
    result = comparison.compare_reports(rep_a, rep_b)       # REAL embedding model
    demo_ran += 1
    print(f"\n--- {rep_a['topic']}  vs  {rep_b['topic']} ---")
    print(f"similarity method: {result['themes']['method']}")
    for line in result["summary"]:
        print("  *", line)
    if result["papers"]["shared"]:
        print("  shared papers:")
        for s in result["papers"]["shared"][:5]:
            print("     -", s["title"][:80])
    print(f"  shared key terms: {result['key_terms']['shared']}")
    print("  theme alignment (A -> closest B):")
    for al_ in result["themes"]["alignments"]:
        print(f"     A{al_['a_theme']} -> B{al_['b_theme']}  similarity {al_['similarity']}  [{al_['relation']}{', mutual' if al_['mutual'] else ''}]")
    check(f"{slug_a} vs {slug_b}: result is JSON-serialisable", bool(json.dumps(result)))

check("at least one real pair was compared", demo_ran >= 1)

print("\n" + "=" * 70)
print(f"RESULT: {checks_passed} passed, {checks_failed} failed")
if checks_failed == 0:
    print("All comparison.py checks passed.")