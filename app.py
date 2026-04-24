"""
Flask web API and static UI for the Academic Tutor chatbot.
"""

from __future__ import annotations

import errno
import os
import sys
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from flask import Flask, jsonify, request, send_from_directory

print("[INFO] Importing app (TensorFlow / model load may take 20-60 seconds the first time)...", flush=True)

from chatbot import (
    SUBJECT_NAMES,
    SUBJECT_CLARIFY_OPTIONS,
    chat_with_debug,
    is_ready,
    load_artifacts,
)
from learning_memory import (
    add_candidate_fact,
    add_feedback,
    init_memory_db,
    list_candidate_facts,
    list_verified_facts,
    log_interaction,
    promote_candidate_fact,
    retrieve_verified_answer,
    verify_candidate_fact,
)

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"

app = Flask(__name__, static_folder=str(STATIC), static_url_path="")
MEMORY_DB = ROOT / "learning_memory.db"
init_memory_db(MEMORY_DB)
print(f"[INFO] Learning memory ready: {MEMORY_DB.name}")

_CHAT_STATE_LOCK = threading.Lock()
_CHAT_STATE_BY_CLIENT: Dict[str, Dict[str, Any]] = {}

_startup_err = load_artifacts(ROOT)
if _startup_err:
    print(f"[WARNING] {_startup_err}")
else:
    print("[INFO] Academic Tutor model ready.")


@app.route("/")
def index():
    return send_from_directory(STATIC, "index.html")


@app.route("/api/health")
def health():
    return jsonify(
        {
            "ok": is_ready(),
            "subjects": list(SUBJECT_NAMES.values()),
        }
    )


def _client_key() -> str:
    ip = (request.headers.get("X-Forwarded-For") or request.remote_addr or "unknown").split(",")[0].strip()
    agent = (request.headers.get("User-Agent") or "unknown")[:120]
    return f"{ip}|{agent}"


def _fetch_client_state(client_key: str) -> Optional[Dict[str, Any]]:
    with _CHAT_STATE_LOCK:
        state = _CHAT_STATE_BY_CLIENT.get(client_key)
        return dict(state) if state else None


def _store_client_state(client_key: str, state: Dict[str, Any]) -> None:
    with _CHAT_STATE_LOCK:
        _CHAT_STATE_BY_CLIENT[client_key] = dict(state)


def _should_store_as_topic_turn(subject: str, model_route: Optional[str]) -> bool:
    if subject in {"greeting", "out_of_scope"}:
        return False
    if model_route in {
        "conversation_repair",
        "capabilities_scope",
        "greeting_smalltalk",
        "reasoning_clarify",
    }:
        return False
    return True


def _resolve_message(message: str, chat_context: Optional[Dict[str, Any]] = None):
    mem = retrieve_verified_answer(message)
    if mem is not None:
        return {
            "response": mem["answer"],
            "subject": mem["subject"],
            "confidence": float(mem["score"]),
            "retrieval_source": "memory_verified",
            "debug_payload": None,
            "memory_match": {
                "subject": mem["subject"],
                "score": round(float(mem["score"]), 4),
                "id": mem["id"],
            },
            "follow_up_options": None,
            "follow_up_prompts": None,
        }

    response, subject, confidence, debug_payload = chat_with_debug(message, chat_context)
    follow_up_options = None
    follow_up_prompts = None
    if debug_payload and debug_payload.get("route") == "subject_clarify_mode":
        follow_up_options = SUBJECT_CLARIFY_OPTIONS.get(subject, [])
        label = SUBJECT_NAMES.get(subject, subject)
        follow_up_prompts = [f"In {label}, explain {opt}." for opt in follow_up_options]
    elif debug_payload and debug_payload.get("route") == "reasoning_clarify":
        label = SUBJECT_NAMES.get(subject, subject)
        follow_up_options = [
            "Name the concept (one sentence)",
            "Quote the part of my last answer you mean",
            f"If you meant {label}, add one concrete noun from that field",
        ]
        follow_up_prompts = [
            "I mean [topic]: [your question in full].",
            "About your previous answer, this part: \"...\"",
            f"In {label}, explain [specific term].",
        ]

    return {
        "response": response,
        "subject": subject,
        "confidence": confidence,
        "retrieval_source": "model",
        "debug_payload": debug_payload,
        "memory_match": None,
        "follow_up_options": follow_up_options,
        "follow_up_prompts": follow_up_prompts,
    }


@app.route("/api/chat", methods=["POST"])
def api_chat():
    if not is_ready():
        return (
            jsonify(
                {
                    "error": "Model not loaded. Run `python train.py` in the project folder, then restart the server.",
                }
            ),
            503,
        )
    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()
    debug_mode = bool(data.get("debug", False))
    if not message:
        return jsonify({"error": "Send a non-empty `message` in the JSON body."}), 400

    client_key = _client_key()
    prior_state = _fetch_client_state(client_key)
    chat_ctx: Optional[Dict[str, Any]] = None
    if prior_state:
        chat_ctx = {
            "last_route": prior_state.get("last_route"),
            "last_clarify_thread": prior_state.get("last_clarify_thread"),
            "last_subject": prior_state.get("last_subject"),
        }
    resolved = _resolve_message(message, chat_ctx)
    response = resolved["response"]
    subject = resolved["subject"]
    confidence = resolved["confidence"]
    retrieval_source = resolved["retrieval_source"]
    debug_payload = resolved["debug_payload"]
    follow_up_options = resolved["follow_up_options"]
    follow_up_prompts = resolved["follow_up_prompts"]
    memory_match = resolved["memory_match"]

    # If user indicates the previous answer was wrong, retry the last topical question.
    repair_applied = False
    repair_original_question = None
    if (
        debug_payload is not None
        and debug_payload.get("route") == "conversation_repair"
        and prior_state
        and prior_state.get("last_topic_question")
    ):
        repair_original_question = str(prior_state["last_topic_question"])
        retried = _resolve_message(repair_original_question, None)
        response = (
            f"Re-reading your earlier question:\n\"{repair_original_question}\"\n\n"
            f"{retried['response']}"
        )
        subject = retried["subject"]
        confidence = retried["confidence"]
        retrieval_source = f"{retried['retrieval_source']}_repair_retry"
        follow_up_options = retried["follow_up_options"]
        follow_up_prompts = retried["follow_up_prompts"]
        memory_match = retried["memory_match"]
        repair_applied = True
        if debug_payload is not None:
            debug_payload["repair_replayed_question"] = repair_original_question
            debug_payload["repair_replay_subject"] = subject
            debug_payload["repair_replay_source"] = retrieval_source

    interaction_id = log_interaction(
        user_message=message,
        bot_reply=response,
        subject_key=subject,
        confidence=confidence,
        retrieval_source=retrieval_source,
    )

    label = (
        "Out of scope"
        if subject == "out_of_scope"
        else SUBJECT_NAMES.get(subject, subject)
    )
    payload = {
        "reply": response,
        "subject": label,
        "subject_key": subject,
        "confidence": round(confidence, 4),
        "confidence_band": (
            (debug_payload or {}).get("confidence_band")
            if isinstance(debug_payload, dict)
            else None
        ),
        "retrieval_source": retrieval_source,
        "interaction_id": interaction_id,
        "follow_up_options": follow_up_options,
        "follow_up_prompts": follow_up_prompts,
        "repair_applied": repair_applied,
        "repair_original_question": repair_original_question,
    }

    model_route = debug_payload.get("route") if isinstance(debug_payload, dict) else None
    if repair_applied and repair_original_question:
        _store_client_state(
            client_key,
            {
                "last_topic_question": repair_original_question,
                "last_subject": subject,
                "last_response": response,
                "last_retrieval_source": retrieval_source,
                "last_interaction_id": interaction_id,
                "last_route": None,
                "last_clarify_thread": None,
            },
        )
    elif model_route == "reasoning_clarify":
        prev = (prior_state or {}).get("last_clarify_thread") if (prior_state or {}).get("last_route") == "reasoning_clarify" else ""
        thread = f"{prev} {message}".strip()
        if len(thread) > 1200:
            thread = thread[-1200:]
        merged_r = dict(prior_state or {})
        merged_r.update(
            {
                "last_route": "reasoning_clarify",
                "last_clarify_thread": thread,
                "last_subject": subject,
                "last_response": response,
                "last_retrieval_source": retrieval_source,
                "last_interaction_id": interaction_id,
            }
        )
        _store_client_state(client_key, merged_r)
    elif model_route in {
        "greeting_smalltalk",
        "capabilities_scope",
        "conversation_repair",
        "scope_gate_off_topic",
    }:
        merged = dict(prior_state or {})
        merged.update(
            {
                "last_route": None,
                "last_clarify_thread": None,
            }
        )
        _store_client_state(client_key, merged)
    elif _should_store_as_topic_turn(subject, model_route):
        merged_t = dict(prior_state or {})
        merged_t.update(
            {
                "last_topic_question": message,
                "last_subject": subject,
                "last_response": response,
                "last_retrieval_source": retrieval_source,
                "last_interaction_id": interaction_id,
                "last_route": None,
                "last_clarify_thread": None,
            }
        )
        _store_client_state(client_key, merged_t)

    if debug_mode:
        payload["debug"] = {
            "requested": True,
            "memory_hit": retrieval_source.startswith("memory_verified"),
            "memory_match": memory_match,
            "model_debug": debug_payload,
            "repair": {
                "applied": repair_applied,
                "replayed_question": repair_original_question,
            },
        }
    return jsonify(payload)


@app.route("/api/learn/candidate", methods=["POST"])
def api_learn_candidate():
    data = request.get_json(silent=True) or {}
    subject = (data.get("subject") or "").strip()
    prompt = (data.get("prompt") or "").strip()
    answer = (data.get("answer") or "").strip()
    source = (data.get("source") or "user").strip()
    submitted_by = (data.get("submitted_by") or "anonymous").strip()

    allowed_subjects = [k for k in SUBJECT_NAMES.keys() if k != "greeting"]
    if subject not in allowed_subjects:
        return jsonify({"error": f"subject must be one of {allowed_subjects}"}), 400
    if len(prompt) < 8:
        return jsonify({"error": "prompt is too short (min 8 chars)."}), 400
    if len(answer) < 20:
        return jsonify({"error": "answer is too short (min 20 chars)."}), 400

    out = add_candidate_fact(subject, prompt, answer, source, submitted_by)
    return jsonify({"ok": True, **out})


@app.route("/api/learn/pending", methods=["GET"])
def api_learn_pending():
    limit = int(request.args.get("limit", "50"))
    return jsonify({"ok": True, "items": list_candidate_facts(status="pending", limit=limit)})


@app.route("/api/learn/verify", methods=["POST"])
def api_learn_verify():
    data = request.get_json(silent=True) or {}
    candidate_id = int(data.get("candidate_id") or 0)
    if candidate_id <= 0:
        return jsonify({"error": "candidate_id must be a positive integer."}), 400
    allowed_subjects = [k for k in SUBJECT_NAMES.keys() if k != "greeting"]
    out = verify_candidate_fact(candidate_id, allowed_subjects)
    if not out.get("ok"):
        return jsonify(out), 400
    return jsonify(out)


@app.route("/api/learn/promote", methods=["POST"])
def api_learn_promote():
    data = request.get_json(silent=True) or {}
    candidate_id = int(data.get("candidate_id") or 0)
    if candidate_id <= 0:
        return jsonify({"error": "candidate_id must be a positive integer."}), 400
    out = promote_candidate_fact(candidate_id)
    if not out.get("ok"):
        return jsonify(out), 400
    return jsonify(out)


@app.route("/api/learn/verified", methods=["GET"])
def api_learn_verified():
    subject = request.args.get("subject")
    limit = int(request.args.get("limit", "100"))
    return jsonify({"ok": True, "items": list_verified_facts(subject=subject, limit=limit)})


@app.route("/api/learn/feedback", methods=["POST"])
def api_learn_feedback():
    data = request.get_json(silent=True) or {}
    interaction_id = int(data.get("interaction_id") or 0)
    rating = int(data.get("rating") if data.get("rating") is not None else 0)
    notes = (data.get("notes") or "").strip()
    if interaction_id <= 0:
        return jsonify({"error": "interaction_id must be a positive integer."}), 400
    out = add_feedback(interaction_id=interaction_id, rating=rating, notes=notes)
    if not out.get("ok"):
        return jsonify(out), 400
    return jsonify(out)


def create_app():
    return app


def _addr_in_use(err: OSError) -> bool:
    if getattr(err, "winerror", None) == 10048:  # Windows: WSAEADDRINUSE
        return True
    e = err.errno
    if e == errno.EADDRINUSE or e == 10048 or e == 48:
        return True
    msg = str(err).lower()
    return "address" in msg and "use" in msg


if __name__ == "__main__":
    # 127.0.0.1 is more reliable on some Windows setups than 0.0.0.0; use FLASK_HOST=0.0.0.0 for LAN access
    host = os.environ.get("FLASK_HOST", "127.0.0.1")
    start_port = int(os.environ.get("PORT", "5000"))
    print(f'[INFO] Host={host!r} (set FLASK_HOST=0.0.0.0 to listen on all interfaces).', flush=True)
    for port in range(start_port, start_port + 8):
        try:
            print(f"[INFO] Starting server: http://{host}:{port}/  (Ctrl+C to stop)\n", flush=True)
            app.run(
                host=host,
                port=port,
                debug=False,
                use_reloader=False,
                threaded=True,
            )
            break
        except OSError as e:
            if _addr_in_use(e):
                print(f"[WARNING] Port {port} is in use, trying {port + 1}... ", flush=True)
                continue
            print(f"[ERROR] Could not start server: {e}", file=sys.stderr, flush=True)
            raise
    else:
        print(
            "[ERROR] No free port in range. Set PORT=8080 or close apps using 5000-5007 (e.g. other dev servers).",
            file=sys.stderr,
            flush=True,
        )
        sys.exit(1)
