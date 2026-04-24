"""
Persistent learning/verification memory for the chatbot.
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

_DB_PATH: Optional[Path] = None
_LOCK = threading.Lock()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


_STOPWORDS = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "was",
    "were",
    "be",
    "to",
    "of",
    "in",
    "on",
    "at",
    "for",
    "and",
    "or",
    "with",
    "what",
    "who",
    "when",
    "where",
    "why",
    "how",
    "can",
    "you",
    "explain",
    "define",
    "tell",
    "me",
    "about",
}


def _content_tokens(text: str) -> set:
    tokens = {t for t in _normalize(text).split() if len(t) >= 3 and t not in _STOPWORDS}
    return tokens


def _token_overlap_ratio(a: str, b: str) -> float:
    ta = _content_tokens(a)
    tb = _content_tokens(b)
    if not ta or not tb:
        return 0.0
    inter = ta.intersection(tb)
    return float(len(inter) / max(1, min(len(ta), len(tb))))


def _conn() -> sqlite3.Connection:
    if _DB_PATH is None:
        raise RuntimeError("Memory DB not initialized.")
    c = sqlite3.connect(str(_DB_PATH))
    c.row_factory = sqlite3.Row
    return c


def init_memory_db(db_path: Path) -> None:
    global _DB_PATH
    _DB_PATH = db_path
    with _conn() as c:
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS verified_facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject TEXT NOT NULL,
                prompt TEXT NOT NULL,
                answer TEXT NOT NULL,
                source TEXT DEFAULT 'manual',
                confidence REAL DEFAULT 1.0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                last_used_at TEXT,
                use_count INTEGER DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_verified_subject ON verified_facts(subject);

            CREATE TABLE IF NOT EXISTS candidate_facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject TEXT NOT NULL,
                prompt TEXT NOT NULL,
                answer TEXT NOT NULL,
                source TEXT DEFAULT 'user',
                submitted_by TEXT DEFAULT 'anonymous',
                status TEXT NOT NULL DEFAULT 'pending', -- pending, verified, rejected, promoted
                verification_score REAL DEFAULT 0.0,
                verification_notes TEXT DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_candidate_status ON candidate_facts(status);

            CREATE TABLE IF NOT EXISTS interaction_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_message TEXT NOT NULL,
                bot_reply TEXT NOT NULL,
                subject_key TEXT NOT NULL,
                confidence REAL NOT NULL,
                retrieval_source TEXT NOT NULL, -- model, memory_verified, fallback
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS interaction_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                interaction_id INTEGER NOT NULL,
                rating INTEGER NOT NULL, -- -1, 0, 1
                notes TEXT DEFAULT '',
                created_at TEXT NOT NULL,
                FOREIGN KEY(interaction_id) REFERENCES interaction_logs(id)
            );
            """
        )


def add_candidate_fact(
    subject: str, prompt: str, answer: str, source: str = "user", submitted_by: str = "anonymous"
) -> Dict:
    now = _utc_now()
    with _LOCK, _conn() as c:
        cur = c.execute(
            """
            INSERT INTO candidate_facts(subject, prompt, answer, source, submitted_by, status, created_at, updated_at)
            VALUES(?, ?, ?, ?, ?, 'pending', ?, ?)
            """,
            (subject.strip(), prompt.strip(), answer.strip(), source.strip(), submitted_by.strip(), now, now),
        )
        cid = int(cur.lastrowid)
    return {"id": cid, "status": "pending"}


def list_candidate_facts(status: str = "pending", limit: int = 50) -> List[Dict]:
    with _conn() as c:
        rows = c.execute(
            """
            SELECT id, subject, prompt, answer, source, submitted_by, status,
                   verification_score, verification_notes, created_at, updated_at
            FROM candidate_facts
            WHERE status = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (status, max(1, min(limit, 500))),
        ).fetchall()
    return [dict(r) for r in rows]


def _retrieval_score(prompt: str, subject: str) -> float:
    """Score candidate prompt against existing verified prompts."""
    p = _normalize(prompt)
    if not p:
        return 0.0
    with _conn() as c:
        rows = c.execute(
            "SELECT prompt FROM verified_facts WHERE subject = ? ORDER BY id DESC LIMIT 200", (subject,)
        ).fetchall()
        if not rows:
            rows = c.execute("SELECT prompt FROM verified_facts ORDER BY id DESC LIMIT 200").fetchall()
    corpus = [_normalize(r["prompt"]) for r in rows if _normalize(r["prompt"])]
    if not corpus:
        return 0.0
    vec_word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1)
    vec_char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
    mat_w = vec_word.fit_transform(corpus)
    mat_c = vec_char.fit_transform(corpus)
    q_w = vec_word.transform([p])
    q_c = vec_char.transform([p])
    sw = (q_w @ mat_w.T).toarray()[0]
    sc = (q_c @ mat_c.T).toarray()[0]
    return float(np.max((0.65 * sw) + (0.35 * sc)))


def verify_candidate_fact(candidate_id: int, allowed_subjects: List[str]) -> Dict:
    with _LOCK, _conn() as c:
        row = c.execute(
            "SELECT id, subject, prompt, answer, status FROM candidate_facts WHERE id = ?",
            (candidate_id,),
        ).fetchone()
        if row is None:
            return {"ok": False, "error": "Candidate not found."}
        if row["status"] in ("rejected", "promoted"):
            return {"ok": False, "error": f"Candidate already {row['status']}."}

        subject = (row["subject"] or "").strip()
        prompt = (row["prompt"] or "").strip()
        answer = (row["answer"] or "").strip()
        notes = []
        score = 0.0

        if subject not in allowed_subjects:
            notes.append("Invalid subject.")
        else:
            score += 0.20
        if len(prompt) >= 8:
            score += 0.15
        else:
            notes.append("Prompt is too short.")
        if len(answer) >= 40:
            score += 0.20
        else:
            notes.append("Answer is too short.")

        dup = c.execute(
            """
            SELECT id FROM verified_facts
            WHERE lower(subject)=lower(?) AND lower(prompt)=lower(?)
            LIMIT 1
            """,
            (subject, prompt),
        ).fetchone()
        if dup:
            score += 0.30
            notes.append("Matches existing verified prompt (update possible).")
        else:
            score += min(0.45, _retrieval_score(prompt, subject))

        verified = score >= 0.55 and subject in allowed_subjects and len(answer) >= 40
        new_status = "verified" if verified else "pending"
        c.execute(
            """
            UPDATE candidate_facts
            SET status = ?, verification_score = ?, verification_notes = ?, updated_at = ?
            WHERE id = ?
            """,
            (new_status, score, " ".join(notes).strip(), _utc_now(), candidate_id),
        )

    return {"ok": True, "id": candidate_id, "status": new_status, "verification_score": round(score, 4), "notes": notes}


def promote_candidate_fact(candidate_id: int) -> Dict:
    with _LOCK, _conn() as c:
        row = c.execute(
            """
            SELECT id, subject, prompt, answer, source, status
            FROM candidate_facts WHERE id = ?
            """,
            (candidate_id,),
        ).fetchone()
        if row is None:
            return {"ok": False, "error": "Candidate not found."}
        if row["status"] not in ("verified", "pending"):
            return {"ok": False, "error": f"Candidate status {row['status']} cannot be promoted."}

        now = _utc_now()
        existing = c.execute(
            """
            SELECT id FROM verified_facts
            WHERE lower(subject)=lower(?) AND lower(prompt)=lower(?)
            LIMIT 1
            """,
            (row["subject"], row["prompt"]),
        ).fetchone()
        if existing:
            fid = int(existing["id"])
            c.execute(
                """
                UPDATE verified_facts
                SET answer = ?, source = ?, confidence = ?, updated_at = ?
                WHERE id = ?
                """,
                (row["answer"], row["source"], 0.95, now, fid),
            )
        else:
            cur = c.execute(
                """
                INSERT INTO verified_facts(subject, prompt, answer, source, confidence, created_at, updated_at, use_count)
                VALUES(?, ?, ?, ?, ?, ?, ?, 0)
                """,
                (row["subject"], row["prompt"], row["answer"], row["source"], 0.95, now, now),
            )
            fid = int(cur.lastrowid)

        c.execute(
            "UPDATE candidate_facts SET status = 'promoted', updated_at = ? WHERE id = ?",
            (now, candidate_id),
        )
    return {"ok": True, "candidate_id": candidate_id, "verified_fact_id": fid}


def list_verified_facts(subject: Optional[str] = None, limit: int = 100) -> List[Dict]:
    with _conn() as c:
        if subject:
            rows = c.execute(
                """
                SELECT id, subject, prompt, answer, source, confidence, use_count, updated_at
                FROM verified_facts
                WHERE subject = ?
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (subject, max(1, min(limit, 500))),
            ).fetchall()
        else:
            rows = c.execute(
                """
                SELECT id, subject, prompt, answer, source, confidence, use_count, updated_at
                FROM verified_facts
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (max(1, min(limit, 500)),),
            ).fetchall()
    return [dict(r) for r in rows]


def retrieve_verified_answer(question: str, subject_hint: Optional[str] = None) -> Optional[Dict]:
    q = _normalize(question)
    if not q:
        return None
    with _conn() as c:
        rows = c.execute(
            """
            SELECT id, subject, prompt, answer, confidence
            FROM verified_facts
            ORDER BY updated_at DESC
            LIMIT 500
            """
        ).fetchall()
    if not rows:
        return None

    corpus = [_normalize(r["prompt"]) for r in rows]
    vec_word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1)
    vec_char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
    mat_w = vec_word.fit_transform(corpus)
    mat_c = vec_char.fit_transform(corpus)
    q_w = vec_word.transform([q])
    q_c = vec_char.transform([q])
    sw = (q_w @ mat_w.T).toarray()[0]
    sc = (q_c @ mat_c.T).toarray()[0]
    combo = (0.65 * sw) + (0.35 * sc)

    idx = int(np.argmax(combo))
    score = float(combo[idx])
    best = rows[idx]

    if subject_hint:
        candidates = [(i, float(combo[i])) for i, r in enumerate(rows) if r["subject"] == subject_hint]
        if candidates:
            i2, s2 = max(candidates, key=lambda x: x[1])
            if s2 >= score * 0.9:
                idx, score, best = i2, s2, rows[i2]

    overlap = _token_overlap_ratio(q, str(best["prompt"]))
    # Require both semantic signal and lexical overlap to avoid false memory hits.
    if score < 0.52 or overlap < 0.34:
        return None

    touch_verified_fact(int(best["id"]))
    return {
        "id": int(best["id"]),
        "subject": str(best["subject"]),
        "answer": str(best["answer"]),
        "score": score,
        "overlap": overlap,
        "confidence": float(best["confidence"]),
    }


def touch_verified_fact(fact_id: int) -> None:
    with _LOCK, _conn() as c:
        c.execute(
            """
            UPDATE verified_facts
            SET use_count = use_count + 1, last_used_at = ?, updated_at = ?
            WHERE id = ?
            """,
            (_utc_now(), _utc_now(), fact_id),
        )


def log_interaction(
    user_message: str,
    bot_reply: str,
    subject_key: str,
    confidence: float,
    retrieval_source: str,
) -> int:
    with _LOCK, _conn() as c:
        cur = c.execute(
            """
            INSERT INTO interaction_logs(user_message, bot_reply, subject_key, confidence, retrieval_source, created_at)
            VALUES(?, ?, ?, ?, ?, ?)
            """,
            (user_message, bot_reply, subject_key, float(confidence), retrieval_source, _utc_now()),
        )
        return int(cur.lastrowid)


def add_feedback(interaction_id: int, rating: int, notes: str = "") -> Dict:
    if rating not in (-1, 0, 1):
        return {"ok": False, "error": "rating must be one of -1, 0, 1."}
    with _LOCK, _conn() as c:
        row = c.execute("SELECT id FROM interaction_logs WHERE id = ?", (interaction_id,)).fetchone()
        if row is None:
            return {"ok": False, "error": "interaction_id not found."}
        cur = c.execute(
            """
            INSERT INTO interaction_feedback(interaction_id, rating, notes, created_at)
            VALUES(?, ?, ?, ?)
            """,
            (interaction_id, rating, notes.strip(), _utc_now()),
        )
    return {"ok": True, "id": int(cur.lastrowid)}
