"""
app.py
Desktop app shell for the Literature Review pipeline. Runs a local
Flask server (runs the LangGraph pipeline in graph_pipeline.py -
retriever, synthesizer, critic, hypothesis generator, novelty checker
and evaluator - in a background thread) and opens it in a native
desktop window via pywebview - no
browser tab, no address bar, looks and behaves like a real desktop
app.

Accepts a `settings` object from the frontend's Settings Panel (see
config.DEFAULT_SETTINGS) and threads it through every pipeline
stage, so a researcher's chosen paper count, year range, relevance
threshold, open-access filter, and hypothesis count are respected
end to end. Also exposes an export endpoint that turns the most
recently completed result into a Word or PDF report via export.py.

Every finished run is also saved to the Search History (history.py),
and the /api/history endpoints let the frontend list past searches,
re-open any of them instantly (no re-fetching, no re-running the
models - the saved report is simply loaded back in as the "current
result", so Export Word/PDF works on it too), delete one, or clear
them all. Two saved searches can also be compared side by side
(/api/compare, powered by comparison.py) - this works purely from the
saved reports, so it is instant and re-runs nothing.
"""

import os
import time
import threading
import traceback

import webview
from flask import Flask, jsonify, request, send_from_directory

import config
import history
import comparison
import evaluator
import graph_pipeline
from export import export_to_docx, export_to_pdf

# Load every agent (and with it the language models) once, now, at start-up -
# not during the first search, which would make that search feel stuck.
graph_pipeline.preload_agents()

app = Flask(__name__, static_folder="static", static_url_path="")

STATE_LOCK = threading.Lock()
STATE = {
    "running": False,
    "topic": None,
    "settings": None,
    "stages": {
        "retriever": {"status": "idle", "message": ""},
        "synthesizer": {"status": "idle", "message": ""},
        "critic": {"status": "idle", "message": ""},
        "hypothesis": {"status": "idle", "message": ""},
        "novelty": {"status": "idle", "message": ""},
        "evaluator": {"status": "idle", "message": ""},
    },
    "result": None,
    "error": None,
}


def _reset_state(topic, settings):
    with STATE_LOCK:
        STATE["running"] = True
        STATE["topic"] = topic
        STATE["settings"] = settings
        STATE["result"] = None
        STATE["error"] = None
        for s in STATE["stages"].values():
            s["status"] = "idle"
            s["message"] = ""


def _set_stage(name, status, message=""):
    with STATE_LOCK:
        STATE["stages"][name]["status"] = status
        STATE["stages"][name]["message"] = message


def _on_progress(stage, status, message):
    """Receives live progress from the LangGraph pipeline and publishes it
    to the frontend. Stages the UI does not show (for example the optional
    discrimination test) are ignored."""
    if stage in STATE["stages"]:
        _set_stage(stage, status, message)


def _ensure_evaluation(report):
    """
    Runs saved before the Evaluator existed have no scores. Score them on
    the spot (local and instant) so every run can show its evaluation.
    A failure here never breaks the request.
    """
    if report.get("evaluation") is None:
        try:
            report["evaluation"] = evaluator.evaluate(report)
        except Exception:
            traceback.print_exc()
    return report


def _run_pipeline_background(topic, settings, force_fresh):
    try:
        state = graph_pipeline.run_graph(
            topic,
            settings=settings,
            force_fresh=force_fresh,
            run_discrimination=False,
            progress=_on_progress,
        )

        # The graph stops early (without raising) when too few papers were found.
        if state.get("error"):
            with STATE_LOCK:
                STATE["error"] = state["error"]
                STATE["running"] = False
            return

        final_report = state["final_report"]

        # Save to Search History. This is wrapped in its own try/except
        # so that a history problem (disk full, permissions...) can
        # never turn a successfully finished run into an error.
        try:
            saved_id = history.save_run(final_report)
            print(f"[history] Saved this run to history as '{saved_id}'")
        except Exception as history_error:
            print(f"[history] WARNING: could not save this run to history: {history_error}")

        with STATE_LOCK:
            STATE["result"] = final_report
            STATE["running"] = False

    except Exception as e:
        traceback.print_exc()
        with STATE_LOCK:
            STATE["error"] = str(e)
            STATE["running"] = False


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/settings/defaults")
def api_settings_defaults():
    """Lets the frontend fetch the default settings + bounds once at
    startup, so the Settings Panel's sliders/inputs are always in
    sync with config.py without duplicating the numbers in JS."""
    return jsonify({
        "defaults": config.DEFAULT_SETTINGS,
        "bounds": config.SETTINGS_BOUNDS,
    })


@app.route("/api/run", methods=["POST"])
def api_run():
    data = request.get_json(force=True)
    topic = (data.get("topic") or "").strip()
    force_fresh = bool(data.get("force_fresh", False))
    raw_settings = data.get("settings") or {}

    if not topic:
        return jsonify({"error": "Topic is required"}), 400

    with STATE_LOCK:
        if STATE["running"]:
            return jsonify({"error": "A pipeline run is already in progress"}), 409

    settings = config.merge_settings(raw_settings)

    _reset_state(topic, settings)
    thread = threading.Thread(target=_run_pipeline_background, args=(topic, settings, force_fresh), daemon=True)
    thread.start()

    return jsonify({"started": True, "settings": settings})


@app.route("/api/status")
def api_status():
    with STATE_LOCK:
        return jsonify(STATE)


@app.route("/api/export", methods=["POST"])
def api_export():
    data = request.get_json(force=True)
    fmt = data.get("format")

    if fmt not in ("docx", "pdf"):
        return jsonify({"error": "format must be 'docx' or 'pdf'"}), 400

    with STATE_LOCK:
        result = STATE.get("result")

    if not result:
        return jsonify({"error": "No completed review to export yet. Run the pipeline first."}), 400

    try:
        if fmt == "docx":
            path = export_to_docx(result)
        else:
            path = export_to_pdf(result)
        return jsonify({"path": path, "filename": os.path.basename(path)})
    except Exception as e:
        return jsonify({"error": f"Export failed: {e}"}), 500


@app.route("/api/history", methods=["GET"])
def api_history_list():
    """Lightweight list of every saved search, newest first (no full reports)."""
    return jsonify({"history": history.list_history()})


@app.route("/api/history/<run_id>/open", methods=["POST"])
def api_history_open(run_id):
    """
    Re-opens a saved search. The saved report becomes the app's
    current result (so Export Word/PDF works on it) and is returned
    to the frontend to display. Refused while a pipeline run is in
    progress, so a running job's state is never overwritten.
    """
    entry = history.load_run(run_id)
    if entry is None:
        return jsonify({"error": "That history entry was not found."}), 404

    report = _ensure_evaluation(entry["report"])

    with STATE_LOCK:
        if STATE["running"]:
            return jsonify({"error": "A pipeline run is in progress. Wait for it to finish first."}), 409
        STATE["topic"] = report.get("topic")
        STATE["settings"] = report.get("settings")
        STATE["result"] = report
        STATE["error"] = None
        for stage in STATE["stages"].values():
            stage["status"] = "done"
            stage["message"] = "Loaded from history"

    return jsonify({
        "id": entry["id"],
        "saved_at": entry["saved_at"],
        "settings_label": entry.get("settings_label", ""),
        "report": report,
    })


@app.route("/api/history/<run_id>", methods=["DELETE"])
def api_history_delete(run_id):
    """Deletes one saved search."""
    if history.delete_run(run_id):
        return jsonify({"deleted": run_id})
    return jsonify({"error": "That history entry was not found."}), 404


@app.route("/api/history", methods=["DELETE"])
def api_history_clear():
    """Deletes ALL saved searches (the frontend asks the user to confirm first)."""
    return jsonify({"cleared": history.clear_history()})


@app.route("/api/compare", methods=["POST"])
def api_compare():
    """
    Compares two saved searches. Expects JSON {"a": <history id>, "b": <history id>}.
    Returns both saved entries (so the frontend can show them side by side)
    plus the computed comparison. Refused while a pipeline run is in
    progress, because the comparison uses the same embedding model the
    pipeline is using and the two should not run at the same moment.
    """
    data = request.get_json(force=True, silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Send JSON like {\"a\": \"<id>\", \"b\": \"<id>\"}."}), 400

    id_a, id_b = data.get("a"), data.get("b")
    if not id_a or not id_b:
        return jsonify({"error": "Pick two saved searches to compare."}), 400
    if id_a == id_b:
        return jsonify({"error": "Pick two DIFFERENT saved searches."}), 400

    with STATE_LOCK:
        if STATE["running"]:
            return jsonify({"error": "A pipeline run is in progress. Wait for it to finish, then compare."}), 409

    entry_a, entry_b = history.load_run(id_a), history.load_run(id_b)
    if entry_a is None or entry_b is None:
        return jsonify({"error": "One of those saved searches was not found (it may have been deleted)."}), 404

    _ensure_evaluation(entry_a["report"])
    _ensure_evaluation(entry_b["report"])

    try:
        result = comparison.compare_reports(entry_a["report"], entry_b["report"])
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Comparison failed: {e}"}), 500

    def card(entry):
        return {
            "id": entry["id"],
            "topic": entry["topic"],
            "saved_at": entry["saved_at"],
            "settings_label": entry.get("settings_label", ""),
            "report": entry["report"],
        }

    return jsonify({"a": card(entry_a), "b": card(entry_b), "comparison": result})


def start_flask():
    app.run(port=5000, debug=False, use_reloader=False)


if __name__ == "__main__":
    print("Starting local server and loading models (first launch may take a moment)...")
    flask_thread = threading.Thread(target=start_flask, daemon=True)
    flask_thread.start()

    time.sleep(1.5)  # give Flask a moment to bind before opening the window

    webview.create_window(
        "Literature Review Agent",
        "http://127.0.0.1:5000",
        width=1280,
        height=860,
        min_size=(1000, 650),
    )
    webview.start()