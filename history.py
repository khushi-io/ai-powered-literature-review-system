"""
history.py
Search History for the Literature Review Agent.

Every time the pipeline finishes a run, app.py hands the finished
report to save_run(). The report is stored as ONE JSON file inside
config.HISTORY_DIR, so a researcher can later re-open any past
search instantly - no re-fetching from OpenAlex and no re-running
the models. Nothing here uses the network or any LLM; it is plain
local file storage.

How entries are identified
--------------------------
Each entry's id is built from the topic + the settings used, in the
same spirit as the project's cache keys (see config.settings_cache_suffix):

    <readable topic slug>_<8-character hash of topic + settings>

so running the SAME topic with the SAME settings again simply
UPDATES that entry (and moves it to the top), instead of filling the
history with duplicates. Running the same topic with DIFFERENT
settings creates a separate entry, because it is a genuinely
different search.

Safety notes
------------
* Ids that arrive from the web UI are validated against a strict
  pattern (lowercase letters, digits, underscores only) before they
  are turned into a file path, so a crafted id such as "../secret"
  can never read or delete files outside the history folder.
* Files are written to a temporary name first and then renamed, so
  a crash half-way through writing never leaves a corrupted entry.
* One unreadable/corrupted file never breaks the whole history
  list - it is skipped with a message in the terminal.
"""

import os
import re
import json
import hashlib
import datetime

import config

_ID_PATTERN = re.compile(r"^[a-z0-9_]{1,80}$")

# Friendly names for settings, used to build the short human-readable
# description shown under each history entry in the UI.
_SETTING_LABELS = {
    "max_papers": "max papers",
    "fetch_limit": "fetch limit",
    "min_relevance": "min relevance",
    "year_from": "from year",
    "year_to": "to year",
    "recent_only": "recent only",
    "open_access_only": "open access only",
    "num_hypotheses": "hypotheses",
}


def _history_dir():
    # Read from config at CALL time (not import time) so tests can
    # point history at a temporary folder without touching real data.
    os.makedirs(config.HISTORY_DIR, exist_ok=True)
    return config.HISTORY_DIR


def _normalize_topic(topic):
    return " ".join(str(topic or "").lower().split())


def _slug(topic, max_len=40):
    safe = "".join(c if (c.isascii() and c.isalnum()) else "_" for c in _normalize_topic(topic))
    while "__" in safe:
        safe = safe.replace("__", "_")
    safe = safe.strip("_")[:max_len].strip("_")
    return safe or "topic"


def make_id(topic, settings):
    """
    Deterministic id for a (topic, settings) pair. Same topic + same
    settings always gives the same id; anything different gives a
    different one.
    """
    merged = config.merge_settings(settings)
    key = _normalize_topic(topic) + "|" + json.dumps(merged, sort_keys=True)
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()[:8]
    return f"{_slug(topic)}_{digest}"


def is_valid_id(run_id):
    """True only for ids that are safe to turn into a file name."""
    return isinstance(run_id, str) and bool(_ID_PATTERN.match(run_id))


def _entry_path(run_id):
    return os.path.join(_history_dir(), f"{run_id}.json")


def describe_settings(settings):
    """
    Short text describing how a run's settings differ from the
    defaults, e.g. "max papers 12, open access only". Returns
    "default settings" when nothing was changed.
    """
    merged = config.merge_settings(settings)
    parts = []
    for key, default_value in config.DEFAULT_SETTINGS.items():
        value = merged.get(key)
        if value == default_value:
            continue
        label = _SETTING_LABELS.get(key, key)
        if isinstance(value, bool):
            if value:
                parts.append(label)
        else:
            parts.append(f"{label} {value}")
    return ", ".join(parts) if parts else "default settings"


def _summarize(report):
    """Small set of headline numbers shown on each history card."""
    novelty = report.get("novelty_check") or []
    return {
        "num_papers": report.get("num_papers", len(report.get("papers") or [])),
        "num_themes": len(report.get("themes") or {}),
        "num_hypotheses": len(report.get("hypotheses") or []),
        "num_close_matches": sum(1 for n in novelty if n.get("is_close_match")),
        "critic_approved": bool(report.get("critic_approved")),
        "critic_support_ratio": report.get("critic_support_ratio"),
        # Overall quality score from the Evaluator (None for runs saved
        # before the Evaluator existed - they are scored when re-opened).
        "evaluation_overall": ((report.get("evaluation") or {}).get("overall") or {}).get("score"),
        "evaluation_label": ((report.get("evaluation") or {}).get("overall") or {}).get("label"),
    }


def save_run(report):
    """
    Stores a finished pipeline report as a history entry and returns
    its id. If an entry for the same topic + settings already exists
    it is overwritten (and gets a fresh timestamp).
    """
    topic = report.get("topic", "")
    settings = report.get("settings")
    run_id = make_id(topic, settings)

    entry = {
        "id": run_id,
        "saved_at": datetime.datetime.now().isoformat(timespec="microseconds"),
        "topic": topic,
        "settings": config.merge_settings(settings),
        "settings_label": describe_settings(settings),
        "summary": _summarize(report),
        "report": report,
    }

    final_path = _entry_path(run_id)
    temp_path = final_path + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(entry, f, ensure_ascii=False, indent=2, default=str)
    os.replace(temp_path, final_path)  # atomic swap: never leaves a half-written file
    return run_id


def _read_entry(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_history():
    """
    Returns a list of lightweight records (id, topic, saved_at,
    settings_label, summary) for every saved search, NEWEST FIRST.
    The full report is not included - use load_run() for that.
    """
    folder = _history_dir()
    records = []
    for name in os.listdir(folder):
        if not name.endswith(".json"):
            continue
        path = os.path.join(folder, name)
        try:
            entry = _read_entry(path)
            records.append({
                "id": entry["id"],
                "topic": entry["topic"],
                "saved_at": entry["saved_at"],
                "settings_label": entry.get("settings_label", ""),
                "summary": entry.get("summary", {}),
            })
        except Exception as e:
            print(f"[history] Skipping unreadable history file {name}: {e}")
    # Microsecond timestamps make ties practically impossible; the id
    # is a final tie-breaker so the order is always the same.
    records.sort(key=lambda r: (r["saved_at"], r["id"]), reverse=True)
    return records


def load_run(run_id):
    """
    Returns the full saved entry (including the complete "report")
    for an id, or None if the id is invalid, missing or unreadable.
    """
    if not is_valid_id(run_id):
        return None
    path = _entry_path(run_id)
    if not os.path.exists(path):
        return None
    try:
        return _read_entry(path)
    except Exception as e:
        print(f"[history] Could not read {run_id}: {e}")
        return None


def delete_run(run_id):
    """Deletes one entry. Returns True if something was deleted."""
    if not is_valid_id(run_id):
        return False
    path = _entry_path(run_id)
    if not os.path.exists(path):
        return False
    os.remove(path)
    return True


def clear_history():
    """Deletes every history entry. Returns how many were removed."""
    folder = _history_dir()
    removed = 0
    for name in os.listdir(folder):
        if name.endswith(".json") or name.endswith(".json.tmp"):
            os.remove(os.path.join(folder, name))
            removed += 1
    return removed