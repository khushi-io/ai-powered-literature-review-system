"""
test_history.py
Standalone self-test for history.py. Uses a TEMPORARY folder, so it
never touches your real history. Prints PASS/FAIL for each check.

Run with:  python test_history.py
"""

import os
import json
import tempfile

import config
import history

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


def fake_report(topic, settings=None, hypotheses=3, close_matches=1):
    merged = config.merge_settings(settings)
    return {
        "topic": topic,
        "settings": merged,
        "num_papers": 12,
        "papers": [{"title": "Paper A"}, {"title": "Paper B"}],
        "themes": {"0": {"paper_count": 5}, "1": {"paper_count": 7}},
        "gaps": ["a gap"],
        "contradictions": [],
        "critic_approved": True,
        "critic_iterations": 1,
        "critic_support_ratio": 0.93,
        "hypotheses": [{"id": f"H{i+1}", "hypothesis": "x"} for i in range(hypotheses)],
        "key_terms": ["term"],
        "novelty_check": [{"hypothesis_id": "H1", "is_close_match": i < close_matches}
                          for i in range(hypotheses)],
    }


# Point history at a throw-away folder for the whole test.
temp_folder = tempfile.mkdtemp(prefix="history_test_")
config.HISTORY_DIR = temp_folder
print(f"Using temporary folder: {temp_folder}\n")

print("1. Saving and listing")
check("history starts empty", history.list_history() == [])

id_a = history.save_run(fake_report("Ethical AI"))
check("save_run returns an id", isinstance(id_a, str) and len(id_a) > 0)
check("id is safe (valid pattern)", history.is_valid_id(id_a))
check("one entry listed after one save", len(history.list_history()) == 1)

print("\n2. Same topic + same settings updates instead of duplicating")
id_a2 = history.save_run(fake_report("  ethical   AI  "))   # different spacing / case
check("same id for same topic + settings", id_a2 == id_a)
check("still only one entry", len(history.list_history()) == 1)

print("\n3. Same topic + different settings is a separate entry")
id_b = history.save_run(fake_report("Ethical AI", settings={"max_papers": 12, "open_access_only": True}))
check("different id for different settings", id_b != id_a)
check("two entries now", len(history.list_history()) == 2)

print("\n4. Listing details")
records = history.list_history()
topics = [r["topic"] for r in records]
check("newest first (the open-access run was saved last)", records[0]["id"] == id_b)
check("record has summary numbers", records[0]["summary"]["num_papers"] == 12
      and records[0]["summary"]["num_themes"] == 2
      and records[0]["summary"]["num_hypotheses"] == 3
      and records[0]["summary"]["num_close_matches"] == 1)
check("list does NOT include the heavy full report", "report" not in records[0])
labels = {r["id"]: r["settings_label"] for r in records}
check("default run labelled 'default settings'", labels[id_a] == "default settings")
check("custom run label mentions changes",
      "max papers 12" in labels[id_b] and "open access only" in labels[id_b])
print(f"        label for custom run: '{labels[id_b]}'")

scored = fake_report("Scored topic")
scored["evaluation"] = {"overall": {"score": 0.812, "label": "strong"}}
id_scored = history.save_run(scored)
rec = [r for r in history.list_history() if r["id"] == id_scored][0]
check("evaluation score is stored in the summary", rec["summary"]["evaluation_overall"] == 0.812
      and rec["summary"]["evaluation_label"] == "strong")
check("runs without an evaluation have None, not a crash", records[0]["summary"]["evaluation_overall"] is None)
history.delete_run(id_scored)
records = history.list_history()

print("\n5. Opening a saved run")
entry = history.load_run(id_a)
check("load_run returns the entry", entry is not None and entry["id"] == id_a)
check("full report is inside", entry["report"]["topic"] == "  ethical   AI  "
      or entry["report"]["topic"].strip().lower().split() == ["ethical", "ai"])
check("report round-trips with papers + hypotheses",
      len(entry["report"]["papers"]) == 2 and len(entry["report"]["hypotheses"]) == 3)
check("unknown id returns None", history.load_run("does_not_exist_00000000") is None)

print("\n6. Safety: bad ids are rejected")
for bad in ["../secret", "..\\secret", "a/b", "", None, "UPPER_case", "x" * 200, "id with space"]:
    check(f"rejected id {repr(bad)[:30]}", not history.is_valid_id(bad))
check("load_run refuses path traversal", history.load_run("../config") is None)
check("delete_run refuses path traversal", history.delete_run("../config") is False)

print("\n7. Non-English topic still gets a safe id")
id_ru = history.save_run(fake_report("\u042d\u0442\u0438\u043a\u0430 \u0438\u0441\u043a\u0443\u0441\u0441\u0442\u0432\u0435\u043d\u043d\u043e\u0433\u043e \u0438\u043d\u0442\u0435\u043b\u043b\u0435\u043a\u0442\u0430"))
check("id valid for Cyrillic topic", history.is_valid_id(id_ru))
check("Cyrillic topic survives in the saved file",
      history.load_run(id_ru)["topic"].startswith("\u042d\u0442\u0438\u043a\u0430"))

print("\n8. A corrupted file does not break the list")
with open(os.path.join(temp_folder, "broken_00000000.json"), "w", encoding="utf-8") as f:
    f.write("{ this is not valid json")
before = len(history.list_history())
check("list still works and skips the broken file", before == 3)

print("\n9. Deleting")
check("delete_run returns True for existing entry", history.delete_run(id_b) is True)
check("deleted entry is gone", history.load_run(id_b) is None)
check("delete_run returns False if already gone", history.delete_run(id_b) is False)
check("two good entries remain", len(history.list_history()) == 2)

print("\n10. Clearing everything")
removed = history.clear_history()
check("clear_history removed files", removed >= 2)
check("history is empty afterwards", history.list_history() == [])

print("\n" + "=" * 50)
print(f"RESULT: {checks_passed} passed, {checks_failed} failed")
if checks_failed == 0:
    print("All history.py checks passed.")