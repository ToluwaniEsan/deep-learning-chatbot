"""
Offline evaluation for Academic Tutor routing and subject classification.

Reads a CSV of user messages with optional expected `subject_key` and/or
`route` (from chat_with_debug debug payload), runs the model locally, and
prints pass/fail summary.

Usage:
  python eval_chatbot.py
  python eval_chatbot.py --cases path/to/cases.csv
  python eval_chatbot.py --json   # machine-readable summary
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chatbot import SUBJECT_NAMES, chat_with_debug, load_artifacts


def _norm_key(s: str) -> str:
    s = (s or "").strip().lower().replace(" ", "_")
    if not s:
        return ""
    for key, label in SUBJECT_NAMES.items():
        if key.lower() == s:
            return key
        if label.lower().replace(" ", "_") == s or label.lower() == s.replace("_", " "):
            return key
    return s


def _load_rows(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Cases file not found: {path}")
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("CSV has no header row.")
        rows = []
        for i, row in enumerate(reader, start=2):
            msg = (row.get("message") or row.get("prompt") or "").strip()
            if not msg or msg.startswith("#"):
                continue
            rows.append(
                {
                    "line": i,
                    "message": msg,
                    "expected_subject": (row.get("expected_subject") or row.get("subject") or "").strip(),
                    "expected_route": (row.get("expected_route") or row.get("route") or "").strip(),
                    "note": (row.get("note") or "").strip(),
                }
            )
        return rows


def _run_case(message: str) -> Tuple[str, str, float, str, Dict[str, Any]]:
    reply, subject, confidence, debug = chat_with_debug(message, None)
    route = str(debug.get("route") or "")
    return reply, subject, confidence, route, debug


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate chatbot subject + route against a CSV.")
    ap.add_argument(
        "--cases",
        type=Path,
        default=ROOT / "chat_eval_cases.csv",
        help="CSV with columns: message, expected_subject (optional), expected_route (optional)",
    )
    ap.add_argument("--json", action="store_true", help="Print JSON summary only.")
    args = ap.parse_args()

    err = load_artifacts(ROOT)
    if err:
        print(f"[ERROR] {err}", file=sys.stderr)
        return 2

    try:
        rows = _load_rows(args.cases)
    except (OSError, ValueError) as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 2

    results: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    for row in rows:
        msg = row["message"]
        raw_sub = (row["expected_subject"] or "").strip().lower()
        exp_subjects = [_norm_key(p) for p in raw_sub.split("|") if p.strip()] if raw_sub else []
        raw_route = row["expected_route"].strip().lower() if row["expected_route"] else ""
        exp_routes = [p.strip() for p in raw_route.split("|") if p.strip()] if raw_route else []

        reply, subject, confidence, route, debug = _run_case(msg)

        sub_ok = True if not exp_subjects else (_norm_key(subject) in exp_subjects)
        route_ok = True if not exp_routes else (route.lower() in exp_routes)

        passed = sub_ok and route_ok
        entry = {
            "line": row["line"],
            "message": msg[:120],
            "expected_subject": "|".join(exp_subjects) if exp_subjects else None,
            "expected_route": "|".join(exp_routes) if exp_routes else None,
            "actual_subject": subject,
            "actual_route": route,
            "confidence": round(float(confidence), 4),
            "subject_ok": sub_ok,
            "route_ok": route_ok,
            "passed": passed,
            "note": row.get("note") or "",
        }
        results.append(entry)
        if not passed:
            failures.append(entry)

    total = len(results)
    passed_n = sum(1 for r in results if r["passed"])

    summary = {
        "cases_file": str(args.cases),
        "total": total,
        "passed": passed_n,
        "failed": total - passed_n,
        "pass_rate": round(passed_n / total, 4) if total else 0.0,
        "failures": failures,
    }

    if args.json:
        print(json.dumps(summary, indent=2))
        return 0 if not failures else 1

    print(f"Cases: {args.cases}")
    print(f"Total: {total}  Passed: {passed_n}  Failed: {total - passed_n}  Rate: {summary['pass_rate']:.1%}\n")

    for r in results:
        status = "OK" if r["passed"] else "FAIL"
        parts = [f"[{status}] L{r['line']}: {r['message']!r}"]
        if r["expected_subject"]:
            parts.append(f"subject want={r['expected_subject']!r} got={r['actual_subject']}")
        if r["expected_route"]:
            parts.append(f"route want={r['expected_route']!r} got={r['actual_route']}")
        if not r["passed"]:
            parts.append(f"(sub_ok={r['subject_ok']} route_ok={r['route_ok']})")
        print(" | ".join(parts))

    if failures:
        print("\n--- Failures (detail) ---")
        for r in failures:
            print(json.dumps(r, indent=2))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
