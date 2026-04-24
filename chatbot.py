"""
Deep Learning Academic Tutor Chatbot — Inference Script
Subjects: American History, Math, English, Art, Politics, Biology, Physics,
          Chemistry, Technology & Computing
Level: Middle school through advanced college
Run AFTER train.py has been executed.
"""

from __future__ import annotations

import os
import re
import threading
import csv
import json
from difflib import SequenceMatcher
import numpy as np
import pickle

# Reduce noisy TensorFlow startup logs on Windows CPU runs.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import tensorflow as tf
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tf.get_logger().setLevel("ERROR")

# ── Config (must match train.py) ──────────────────────────────────────────────
MAX_LEN              = 20
CONFIDENCE_THRESHOLD = 0.50
# Slightly stricter TF-IDF gates; ambiguous matches are filtered by lexical overlap (see semantic_retrieval_answer).
SEMANTIC_STRONG_THRESHOLD = 0.43
SEMANTIC_WEAK_THRESHOLD = 0.24
SEMANTIC_TRUST_SCORE = 0.52
SEMANTIC_MIN_LEX_OVERLAP = 0.09
SEMANTIC_MIN_LEX_OVERLAP_STRICT = 0.14
SEMANTIC_SOFT_FLOOR = 0.24

ROOT = Path(__file__).resolve().parent

# Populated by load_artifacts()
model = None
tokenizer = None
le = None
_artifacts_error: Optional[str] = None
_predict_lock = threading.Lock()
_semantic_vectorizer_word: Optional[TfidfVectorizer] = None
_semantic_vectorizer_char: Optional[TfidfVectorizer] = None
_semantic_matrix_word = None
_semantic_matrix_char = None
_semantic_entries: List[Dict[str, str]] = []
_understanding_lexicon: Dict[str, Any] = {}
_subject_alias_vocab: Dict[str, str] = {}
_intent_vocab: Dict[str, List[str]] = {}


def _normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _tokenize_words(text: str) -> List[str]:
    return [w for w in _normalize_text(text).split() if w]


_SEMANTIC_STOP = frozenset(
    {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "to",
        "of",
        "in",
        "on",
        "for",
        "and",
        "or",
        "with",
        "as",
        "at",
        "by",
        "from",
        "that",
        "this",
        "it",
        "its",
        "can",
        "could",
        "would",
        "should",
        "you",
        "your",
        "me",
        "my",
        "we",
        "our",
        "they",
        "their",
        "what",
        "who",
        "when",
        "where",
        "why",
        "how",
        "does",
        "did",
        "do",
        "about",
        "into",
        "like",
        "just",
        "also",
        "very",
        "some",
        "any",
    }
)


def _semantic_content_tokens(text: str) -> set:
    return {t for t in _tokenize_words(text) if len(t) >= 3 and t not in _SEMANTIC_STOP}


def _semantic_lexical_overlap(q_norm: str, prompt_norm: str) -> float:
    """Token overlap between question and retrieval prompt (reduces wrong-fact TF-IDF picks)."""
    ta = _semantic_content_tokens(q_norm)
    tb = _semantic_content_tokens(prompt_norm)
    if not ta or not tb:
        return 0.0
    inter = ta.intersection(tb)
    return float(len(inter) / max(1, min(len(ta), len(tb))))


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def _load_understanding_lexicon(base_dir: Path) -> None:
    """
    Load lexical intent/subject hints used by the question understanding layer.
    """
    global _understanding_lexicon, _subject_alias_vocab, _intent_vocab
    lex_path = base_dir / "question_understanding_lexicon.json"
    if not lex_path.is_file():
        _understanding_lexicon = {}
        _subject_alias_vocab = {}
        _intent_vocab = {}
        return
    try:
        with open(lex_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception as e:
        print(f"[WARNING] Could not load question lexicon: {e}")
        _understanding_lexicon = {}
        _subject_alias_vocab = {}
        _intent_vocab = {}
        return

    _understanding_lexicon = raw if isinstance(raw, dict) else {}
    _subject_alias_vocab = {}
    aliases = _understanding_lexicon.get("subject_aliases", {})
    if isinstance(aliases, dict):
        for subject, values in aliases.items():
            if subject not in SUBJECT_NAMES:
                continue
            if isinstance(values, list):
                for v in values:
                    if isinstance(v, str):
                        vv = _normalize_text(v)
                        if vv:
                            _subject_alias_vocab[vv] = subject

    _intent_vocab = {}
    intents = _understanding_lexicon.get("intent_terms", {})
    if isinstance(intents, dict):
        for intent, values in intents.items():
            if isinstance(values, list):
                cleaned = [_normalize_text(v) for v in values if isinstance(v, str) and _normalize_text(v)]
                if cleaned:
                    _intent_vocab[intent] = cleaned


def detect_subject_hint(text: str) -> Optional[Tuple[str, float, List[str]]]:
    """
    Detect likely target subject from lexical cues (with typo tolerance).
    Returns (subject, score, matched_terms) or None.
    """
    if not text:
        return None
    q_norm = _normalize_text(text)
    if not q_norm:
        return None

    subject_scores: Dict[str, float] = {}
    matched_terms: Dict[str, List[str]] = {}
    tokens = q_norm.split()

    # Phrase-level exact matches first.
    for alias, subject in _subject_alias_vocab.items():
        if alias in q_norm:
            subject_scores[subject] = subject_scores.get(subject, 0.0) + 1.0
            matched_terms.setdefault(subject, []).append(alias)

    # Token-level fuzzy match for typos like "physis" -> "physics".
    vocab_tokens = list(_subject_alias_vocab.keys())
    for tok in tokens:
        best_alias = None
        best_score = 0.0
        for alias in vocab_tokens:
            # Compare token against each alias token.
            for atok in alias.split():
                s = _similarity(tok, atok)
                if s > best_score:
                    best_score = s
                    best_alias = alias
        if best_alias and best_score >= 0.82:
            subj = _subject_alias_vocab[best_alias]
            subject_scores[subj] = subject_scores.get(subj, 0.0) + (best_score * 0.6)
            matched_terms.setdefault(subj, []).append(f"{tok}->{best_alias}")

    if not subject_scores:
        return None

    best_subject, raw_score = max(subject_scores.items(), key=lambda x: x[1])
    # Normalize into [0,1] gently.
    score = min(1.0, raw_score / 2.0)
    return best_subject, score, matched_terms.get(best_subject, [])


def detect_negated_subjects(text: str) -> List[str]:
    q = _normalize_text(text)
    if not q:
        return []
    negated: List[str] = []
    for alias, subj in _subject_alias_vocab.items():
        if re.search(rf"\bnot\s+{re.escape(alias)}\b", q):
            negated.append(subj)
    return list(sorted(set(negated)))


def has_explicit_subject_reference(text: str, subject: str) -> bool:
    q = _normalize_text(text)
    if not q:
        return False
    for alias, subj in _subject_alias_vocab.items():
        if subj != subject:
            continue
        if re.search(rf"\b{re.escape(alias)}\b", q):
            return True
    return False


def detect_intent_hints(text: str) -> List[str]:
    found: List[str] = []
    q = _normalize_text(text)
    if not q:
        return found
    for intent, forms in _intent_vocab.items():
        if any(form in q for form in forms):
            found.append(intent)
    return found


def is_low_information_query(text: str) -> bool:
    q = _normalize_text(text)
    if not q:
        return True
    tokens = [t for t in q.split() if len(t) > 1]
    filler = set(_understanding_lexicon.get("filler_words", [])) if isinstance(_understanding_lexicon, dict) else set()
    filler.update({"this", "that", "it"})
    content = [t for t in tokens if t not in filler]
    return len(content) <= 2


# ── Multi-layer reasoning (underspecified / deictic / meta questions) ─────────
_TOPIC_LEXICON: Optional[set[str]] = None

DEIXIS_PRONOUNS = frozenset(
    {"it", "its", "this", "that", "these", "those", "they", "them", "their", "here"}
)
REASONING_GLUE_WORDS = frozenset(
    {
        "have",
        "has",
        "had",
        "does",
        "did",
        "do",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "a",
        "an",
        "the",
        "any",
        "ever",
        "not",
        "no",
        "yes",
        "you",
        "your",
        "me",
        "my",
        "we",
        "our",
        "us",
        "if",
        "or",
        "so",
        "for",
        "to",
        "of",
        "in",
        "on",
        "about",
        "with",
        "as",
        "at",
        "by",
        "from",
        "into",
        "like",
        "just",
        "also",
        "very",
        "some",
        "can",
        "could",
        "would",
        "should",
        "will",
        "may",
        "might",
        "must",
        "shall",
        "name",
        "names",
        "called",
        "call",
        "mean",
        "means",
        "thing",
        "things",
        "stuff",
        "something",
        "anything",
        "nothing",
        "everything",
        "someone",
        "anyone",
        "question",
        "answer",
    }
)

META_NAMING_PATTERNS = [
    re.compile(r"\b(have|has|had)\s+a\s+name\b"),
    re.compile(r"\b(does|do|did)\s+\S+\s+have\s+a\s+name\b"),
    re.compile(r"\b(if|whether)\s+.+\s+has\s+a\s+name\b", re.I),
    re.compile(r"\bwhat\s+(is|s)\s+it\s+called\b"),
    re.compile(r"\bwhat\s+do\s+you\s+call\s+it\b"),
    re.compile(r"\bany\s+name\s+for\b"),
    re.compile(r"\bofficial\s+name\b"),
    re.compile(r"\bproper\s+name\b"),
    re.compile(r"\bwho\s+named\b"),
]


def _topic_lexicon_tokens() -> set[str]:
    """Tokens drawn from KB keywords (length>=3) to detect a concrete academic anchor."""
    global _TOPIC_LEXICON
    if _TOPIC_LEXICON is not None:
        return _TOPIC_LEXICON
    bag: set[str] = set()
    for subj, kb in KNOWLEDGE_BASE.items():
        if subj in ("greeting",):
            continue
        for kw in kb:
            if kw == "default":
                continue
            for tok in _normalize_text(kw.replace("_", " ")).split():
                if len(tok) >= 3:
                    bag.add(tok)
    _TOPIC_LEXICON = bag
    return bag


# Tokens that appear in KB phrases but are too generic alone to anchor a question (e.g. "name" from "iupac name").
_LEXICON_WEAK_TOKENS = frozenset(
    {
        "name",
        "names",
        "mean",
        "means",
        "type",
        "types",
        "kind",
        "kinds",
        "thing",
        "things",
        "part",
        "parts",
        "way",
        "ways",
        "form",
        "forms",
        "word",
        "words",
        "term",
        "terms",
        "idea",
        "ideas",
        "unit",
        "units",
        "line",
        "lines",
        "side",
        "sides",
        "test",
        "tests",
        "case",
        "cases",
        "mark",
        "marks",
        "work",
        "works",
        "call",
        "use",
        "uses",
    }
)


def _current_turn_has_academic_anchor(user_text: str) -> bool:
    """
    True if THIS user message alone matches a KB topic token or keyword fallback.
    Used after reasoning_clarify so a vague follow-up cannot escape until the user names something concrete.
    """
    if not (user_text or "").strip():
        return False
    qn = _normalize_text(user_text)
    if _merged_hits_topic_lexicon(qn):
        return True
    if keyword_fallback_answer(user_text) is not None:
        return True
    return False


def _merged_hits_topic_lexicon(merged_norm: str) -> bool:
    """True if the question contains a substantive token tied to a known KB topic (not generic glue)."""
    if not merged_norm:
        return False
    lex = _topic_lexicon_tokens()
    for tok in merged_norm.split():
        if len(tok) < 3 or tok not in lex:
            continue
        if tok in _LEXICON_WEAK_TOKENS:
            continue
        return True
    return False


def reasoning_layers_assess(
    raw: str,
    core: str,
    predicted_subject: str,
    subject_hint: Optional[Tuple[str, float, List[str]]],
    negated_subjects: List[str],
    model_conf: float,
    subject_margin: float,
    *,
    pending_followup_clarify: bool = False,
    current_turn_text: str = "",
) -> Dict[str, Any]:
    """
    Layered reasoning on the raw + core question before trusting a default subject answer.

    R0 — Follow-up after reasoning_clarify: keep asking until the *current* message anchors to KB/lexicon.
    R1 — Deixis / thin content: pronouns like *it* / *that* with almost no concrete topic words.
    R2 — Meta-naming without anchor: \"have a name\", \"what is it called\", etc., but no lexicon hit.
    R3 — Overconfident + flat margin: very sure model but almost tied top-2 and no topic anchor.

    `raw` may include prior clarify-thread text for cross-turn pronoun/meta detection; `current_turn_text`
    is always the latest user message only (used for R0 anchor escape).
    """
    merged = f"{raw} {core}".strip()
    merged_norm = _normalize_text(merged)
    layers: Dict[str, Any] = {}
    if not merged_norm:
        return {"clarify": False, "layers": layers, "signals": ["empty"]}

    tokens = [t for t in merged_norm.split() if len(t) > 1]
    filler = set(_understanding_lexicon.get("filler_words", [])) if isinstance(_understanding_lexicon, dict) else set()
    filler.update(REASONING_GLUE_WORDS)

    has_deixis = any(t in DEIXIS_PRONOUNS for t in tokens)
    content = [t for t in tokens if t not in filler and t not in DEIXIS_PRONOUNS]
    lex_hit = _merged_hits_topic_lexicon(merged_norm)
    meta_naming = any(p.search(merged_norm) for p in META_NAMING_PATTERNS)

    layers["R1_deixis_thin"] = bool(has_deixis and len(content) <= 2 and not lex_hit)
    layers["R2_meta_naming_no_anchor"] = bool(meta_naming and not lex_hit)
    layers["R3_overconfident_flat"] = bool(
        model_conf >= 0.90 and subject_margin <= 0.055 and not lex_hit
    )

    ct = (current_turn_text or "").strip()
    # R0 must look at the latest user turn only; do not treat the whole thread as "current" for escaping clarify.
    anchor_src = ct if ct else ("" if pending_followup_clarify else (raw or "").strip())
    anchor_now = _current_turn_has_academic_anchor(anchor_src) if anchor_src else False
    layers["R0_pending_followup"] = bool(pending_followup_clarify)
    layers["R0_current_turn_anchor"] = bool(anchor_now)

    if subject_hint and subject_hint[0] not in negated_subjects:
        if subject_hint[1] >= 0.74 and has_explicit_subject_reference(merged, subject_hint[0]):
            return {"clarify": False, "layers": layers, "signals": ["explicit_subject_hint"]}

    clarify_mid = bool(
        (layers["R1_deixis_thin"] or layers["R2_meta_naming_no_anchor"] or layers["R3_overconfident_flat"])
        and not lex_hit
    )
    if pending_followup_clarify and not anchor_now:
        layers["R0_followup_still_unanchored"] = True
        clarify = True
    elif pending_followup_clarify and anchor_now:
        layers["R0_followup_anchored"] = True
        clarify = clarify_mid
    else:
        clarify = clarify_mid

    signals: List[str] = []
    if layers.get("R0_followup_still_unanchored"):
        signals.append("R0")
    if layers["R1_deixis_thin"]:
        signals.append("R1")
    if layers["R2_meta_naming_no_anchor"]:
        signals.append("R2")
    if layers["R3_overconfident_flat"]:
        signals.append("R3")
    if not clarify:
        signals.append("no_clarify")
    return {"clarify": clarify, "layers": layers, "signals": signals}


def build_reasoning_clarify_response(predicted_subject_key: str) -> str:
    """Human-readable follow-up when the question is too underspecified to answer safely."""
    label = SUBJECT_NAMES.get(predicted_subject_key, predicted_subject_key.replace("_", " ").title())
    subjects_line = ", ".join(_supported_subject_labels())
    return (
        "I parsed your message, but I am **not sure what topic or object you mean** yet — "
        "especially with words like *it*, *that*, or a generic question about a **name** without naming a field.\n\n"
        f"My **best subject guess** from wording alone was **{label}**, but I should not dump a default lesson "
        "there until we anchor the question.\n\n"
        "**Please do one of these:**\n"
        "• Name the **concept** plainly (examples: \"IUPAC name of this compound\", \"DNS hostname vs domain\", "
        "\"Article I of the Constitution\").\n"
        "• If this is a **follow-up** to my last answer, **quote the sentence** you mean or say \"about the previous answer\".\n"
        "• Say which **subject** you want first: "
        f"{subjects_line}.\n\n"
        "Once there is a clear anchor, I will answer in one focused pass."
    )


def _build_semantic_retriever(base_dir: Path) -> None:
    """
    Build a lightweight semantic retriever from:
    1) keyword -> answer pairs in KNOWLEDGE_BASE
    2) question examples in academic_dataset.csv
    """
    global _semantic_vectorizer_word, _semantic_vectorizer_char
    global _semantic_matrix_word, _semantic_matrix_char, _semantic_entries

    entries: List[Dict[str, str]] = []
    seen = set()

    # Keyword prompts from curated KB.
    for subject, kb in KNOWLEDGE_BASE.items():
        subject_name = SUBJECT_NAMES.get(subject, subject).lower()
        for keyword, answer in kb.items():
            if keyword == "default":
                continue
            key = _normalize_text(keyword.replace("_", " "))
            answer_context = _normalize_text(answer)
            for prompt in (
                key,
                f"what is {key}",
                f"who is {key}",
                f"explain {key}",
                f"tell me about {key}",
                f"{subject_name} {key}",
            ):
                sig = (subject, prompt)
                if sig in seen:
                    continue
                seen.add(sig)
                entries.append(
                    {
                        "subject": subject,
                        "prompt": prompt,
                        "text": f"{prompt} {answer_context}",
                        "answer": answer,
                    }
                )

    # Real user-like prompts from dataset (if present).
    dataset_path = base_dir / "academic_dataset.csv"
    if dataset_path.is_file():
        with open(dataset_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                q = _normalize_text(row.get("question", ""))
                subject = row.get("intent", "").strip()
                if not q or subject not in KNOWLEDGE_BASE:
                    continue
                # Map dataset question to best KB answer for that subject.
                answer = None
                for keyword, kb_answer in KNOWLEDGE_BASE[subject].items():
                    if keyword == "default":
                        continue
                    if keyword in q:
                        answer = kb_answer
                        break
                if answer is None:
                    answer = KNOWLEDGE_BASE[subject].get("default", "")
                # Do not index dataset rows that only map to the generic subject default — it
                # poisons TF-IDF retrieval (weak matches return "Good X question! …" instead of facts).
                if not (answer or "").strip() or _is_kb_default_answer(subject, answer):
                    continue
                sig = (subject, q)
                if sig in seen:
                    continue
                seen.add(sig)
                entries.append(
                    {
                        "subject": subject,
                        "prompt": q,
                        "text": f"{q} {_normalize_text(answer)}",
                        "answer": answer,
                    }
                )

    if not entries:
        _semantic_entries = []
        _semantic_vectorizer_word = None
        _semantic_vectorizer_char = None
        _semantic_matrix_word = None
        _semantic_matrix_char = None
        return

    corpus = [e["text"] for e in entries]
    _semantic_vectorizer_word = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        lowercase=True,
        stop_words="english",
        strip_accents="unicode",
        min_df=1,
    )
    _semantic_vectorizer_char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        lowercase=True,
        strip_accents="unicode",
        min_df=1,
    )

    _semantic_matrix_word = _semantic_vectorizer_word.fit_transform(corpus)
    _semantic_matrix_char = _semantic_vectorizer_char.fit_transform(corpus)
    _semantic_entries = entries


def semantic_retrieval_answer(
    question: str, preferred_subject: Optional[str] = None
) -> Optional[Tuple[str, str, float]]:
    """
    Retrieve the closest answer semantically.
    Returns (subject, answer, score) or None if the retriever is unavailable.
    """
    if (
        _semantic_vectorizer_word is None
        or _semantic_vectorizer_char is None
        or _semantic_matrix_word is None
        or _semantic_matrix_char is None
        or not _semantic_entries
    ):
        return None

    q = _normalize_text(question)
    if not q:
        return None

    word_query = _semantic_vectorizer_word.transform([q])
    char_query = _semantic_vectorizer_char.transform([q])
    # TF-IDF vectors are L2-normalized by default, so dot product == cosine similarity.
    score_word = (word_query @ _semantic_matrix_word.T).toarray()[0]
    score_char = (char_query @ _semantic_matrix_char.T).toarray()[0]
    combined = (0.65 * score_word) + (0.35 * score_char)

    order = np.argsort(combined)[::-1]
    top_n = min(14, len(order))

    def _pick_from_order(candidate_order: np.ndarray) -> Optional[Tuple[int, float]]:
        chosen_idx: Optional[int] = None
        chosen_score = -1.0
        for i in candidate_order[:top_n]:
            s = float(combined[int(i)])
            ent = _semantic_entries[int(i)]
            sub_e = str(ent.get("subject") or "")
            ans_e = str(ent.get("answer") or "")
            if _is_kb_default_answer(sub_e, ans_e):
                continue
            prompt = str(ent.get("prompt") or "")
            ov = _semantic_lexical_overlap(q, prompt)
            if s >= SEMANTIC_TRUST_SCORE:
                merit = s
            else:
                if s < 0.34 and ov < SEMANTIC_MIN_LEX_OVERLAP_STRICT:
                    continue
                if s < SEMANTIC_SOFT_FLOOR and ov < SEMANTIC_MIN_LEX_OVERLAP:
                    continue
                if ov < SEMANTIC_MIN_LEX_OVERLAP and s < 0.38:
                    continue
                merit = s + 0.11 * ov
            if merit > chosen_score:
                chosen_score = merit
                chosen_idx = int(i)
        if chosen_idx is None:
            order2 = np.argsort(combined)[::-1]
            for raw_i in order2[: max(top_n, 24)]:
                ent2 = _semantic_entries[int(raw_i)]
                if _is_kb_default_answer(str(ent2.get("subject") or ""), str(ent2.get("answer") or "")):
                    continue
                rs = float(combined[int(raw_i)])
                if rs < 0.30:
                    continue
                raw_ov = _semantic_lexical_overlap(
                    q, str(ent2.get("prompt") or "")
                )
                if raw_ov < 0.08 and rs < 0.44:
                    continue
                return int(raw_i), rs
            return None
        return chosen_idx, float(combined[chosen_idx])

    picked = _pick_from_order(order)
    if picked is None:
        return None
    best_idx, best_score = picked
    best_entry = _semantic_entries[best_idx]

    if preferred_subject:
        preferred_scores = [
            (i, float(combined[i]))
            for i, e in enumerate(_semantic_entries)
            if e["subject"] == preferred_subject
        ]
        if preferred_scores:
            p_idx, p_score = max(preferred_scores, key=lambda x: x[1])
            if p_score >= best_score * 0.90:
                pref_order = np.argsort(combined)[::-1]
                pref_filtered = np.array(
                    [i for i in pref_order if _semantic_entries[int(i)]["subject"] == preferred_subject],
                    dtype=int,
                )
                if pref_filtered.size:
                    pref_pick = _pick_from_order(pref_filtered)
                    if pref_pick is not None:
                        best_idx, best_score = pref_pick
                        best_entry = _semantic_entries[best_idx]

    return best_entry["subject"], best_entry["answer"], best_score


def load_artifacts(base_dir: Optional[Path] = None) -> Optional[str]:
    """
    Load model, tokenizer, and label encoder from base_dir (default: this file's directory).
    Returns None on success, or an error message string on failure.
    """
    global model, tokenizer, le, _artifacts_error
    d = (base_dir or ROOT).resolve()
    try:
        mpath = d / "academic_chatbot_model.keras"
        tpath = d / "tokenizer.pkl"
        lpath = d / "label_encoder.pkl"
        if not mpath.is_file():
            _artifacts_error = f"Model not found: {mpath}. Run train.py first."
            return _artifacts_error
        if not tpath.is_file() or not lpath.is_file():
            _artifacts_error = f"Missing tokenizer or label encoder in {d}. Run train.py first."
            return _artifacts_error
        model = tf.keras.models.load_model(mpath)
        with open(tpath, "rb") as f:
            tokenizer = pickle.load(f)
        with open(lpath, "rb") as f:
            le = pickle.load(f)
        _load_understanding_lexicon(d)
        _build_semantic_retriever(d)
        _artifacts_error = None
        return None
    except Exception as e:
        _artifacts_error = str(e)
        return _artifacts_error


def is_ready() -> bool:
    return model is not None and tokenizer is not None and le is not None

# ── Knowledge Base ────────────────────────────────────────────────────────────
# Each subject maps question keywords → detailed answers
KNOWLEDGE_BASE = {
    "american_history": {
        "first president": "George Washington was the first President of the United States, serving from 1789 to 1797. He is often called the 'Father of His Country' for his leadership during the Revolutionary War and in establishing the new nation.",
        "civil war": "The American Civil War (1861–1865) was fought between the Union (Northern states) and the Confederacy (Southern states). The primary causes included slavery, states' rights, and economic differences. It ended with the Union's victory and the abolition of slavery.",
        "american revolution": "The American Revolution began in 1775 with the Battles of Lexington and Concord. Colonists fought for independence from British rule due to 'taxation without representation' and other grievances. Independence was declared on July 4, 1776.",
        "declaration of independence": "The Declaration of Independence, adopted July 4, 1776, declared the thirteen colonies independent from Britain. Primarily written by Thomas Jefferson, it proclaimed that 'all men are created equal' and listed grievances against King George III.",
        "constitution": "The U.S. Constitution was written in 1787 during the Constitutional Convention in Philadelphia. Key framers included James Madison (known as the 'Father of the Constitution'), Alexander Hamilton, and Benjamin Franklin. It established the framework of American government.",
        "boston tea party": "The Boston Tea Party (December 16, 1773) was a political protest where colonists dumped 342 chests of British tea into Boston Harbor. It was a response to the Tea Act and the principle of taxation without representation, and became a key event leading to the Revolution.",
        "world war": "The United States entered World War I in 1917 after Germany's unrestricted submarine warfare and the Zimmermann Telegram. The U.S. entered World War II on December 8, 1941, the day after Japan's surprise attack on Pearl Harbor.",
        "pearl harbor": "On December 7, 1941, Japan launched a surprise military attack on the U.S. naval base at Pearl Harbor, Hawaii, killing over 2,400 Americans. This led the U.S. to declare war on Japan and enter World War II.",
        "emancipation proclamation": "The Emancipation Proclamation was an executive order issued by President Lincoln on January 1, 1863. It declared that all enslaved people in Confederate states 'shall be then, thenceforward, and forever free,' fundamentally changing the character of the Civil War.",
        "abraham lincoln": "Abraham Lincoln was the 16th President of the United States (1861–1865). He led the country through the Civil War, issued the Emancipation Proclamation, and preserved the Union. He was assassinated by John Wilkes Booth on April 14, 1865.",
        "great depression": "The Great Depression (1929–1939) was the worst economic downturn in modern history, triggered by the stock market crash of October 1929. Unemployment reached 25%, and President FDR responded with the New Deal — a series of programs to provide relief, recovery, and reform.",
        "manifest destiny": "Manifest Destiny was the 19th-century belief that American settlers were destined to expand westward across North America. It drove westward expansion and was used to justify displacing Native American populations and acquiring territories like Texas and California.",
        "trail of tears": "The Trail of Tears (1838–1839) was the forced relocation of the Cherokee Nation and other Native American tribes from their southeastern homelands to present-day Oklahoma. Approximately 4,000 Cherokee died from cold, hunger, and disease during the journey.",
        "founding fathers": "The Founding Fathers were the political leaders who signed the Declaration of Independence or helped draft the U.S. Constitution. Key figures include George Washington, Thomas Jefferson, Benjamin Franklin, John Adams, James Madison, and Alexander Hamilton.",
        "reconstruction": "Reconstruction (1865–1877) was the period after the Civil War when the federal government worked to reintegrate Confederate states and protect the rights of freed Black Americans. It resulted in three Constitutional amendments: the 13th, 14th, and 15th.",
        "cold war": "The Cold War (1947–1991) was a geopolitical tension between the United States and the Soviet Union. While no direct warfare occurred, it involved an arms race, space race, nuclear standoff, and proxy wars in Korea, Vietnam, and elsewhere.",
        "martin luther king": "Dr. Martin Luther King Jr. was a Baptist minister and civil rights leader who used nonviolent protest to fight racial injustice. He delivered the famous 'I Have a Dream' speech in 1963 and won the Nobel Peace Prize in 1964. He was assassinated in Memphis, Tennessee in 1968.",
        "civil rights": "The Civil Rights Movement (1954–1968) was a struggle by Black Americans to end racial segregation and discrimination. Key events include Brown v. Board of Education (1954), the Montgomery Bus Boycott (1955), and the Civil Rights Act of 1964.",
        "louisiana purchase": "The Louisiana Purchase (1803) was a land deal in which the United States bought approximately 828,000 square miles of territory from France for about $15 million. President Thomas Jefferson negotiated the deal, roughly doubling the size of the United States.",
        "george washington": "George Washington was a Virginia planter and military commander who led the Continental Army to victory in the Revolutionary War. He presided over the Constitutional Convention and became the first U.S. President, setting many precedents for the office.",
        "gettysburg": "The Battle of Gettysburg (July 1–3, 1863) was the bloodiest battle of the Civil War, resulting in over 50,000 casualties. The Union victory halted Confederate General Robert E. Lee's invasion of the North and is considered a turning point in the war.",
        "new deal": "The New Deal was a series of programs, public works projects, and financial reforms implemented by President Franklin D. Roosevelt between 1933 and 1939 in response to the Great Depression. It created agencies like the Social Security Administration and the FDIC.",
        "missouri compromise": "The Missouri Compromise (1820) was a federal law that admitted Missouri as a slave state and Maine as a free state, maintaining the balance of power in the Senate. It also prohibited slavery in the Louisiana Territory north of the 36°30′ parallel.",
        "thomas jefferson": "Thomas Jefferson was the primary author of the Declaration of Independence and the 3rd President of the United States (1801–1809). He negotiated the Louisiana Purchase and founded the University of Virginia. He owned enslaved people, a major contradiction to his ideals.",
        "jim crow": "Jim Crow laws were state and local laws enforcing racial segregation in the Southern United States from the 1870s to the 1960s. They mandated 'separate but equal' facilities for Black and white Americans, resulting in deeply unequal treatment and conditions.",
        "vietnam": "The Vietnam War (1955–1975) was a conflict between communist North Vietnam (supported by the Soviet Union and China) and South Vietnam (supported by the United States). The U.S. was directly involved from 1965 to 1973. The war ended with North Vietnam's victory.",
        "default": "Great American History question! I cover middle school through college: colonies through the Constitution, Civil War and Reconstruction, industrialization, world wars, Cold War, and civil rights. Ask about a person, event, or document for a focused answer.",
    },
    "math": {
        "pythagorean": "The Pythagorean Theorem states that in a right triangle, a² + b² = c², where c is the hypotenuse (the side opposite the right angle) and a and b are the other two sides. For example, a 3-4-5 triangle: 3² + 4² = 9 + 16 = 25 = 5².",
        "quadratic": "The quadratic formula is x = (-b ± √(b²-4ac)) / 2a. It solves any equation in the form ax² + bx + c = 0. The discriminant (b²-4ac) tells you the nature of roots: positive = 2 real roots, zero = 1 real root, negative = 2 complex roots.",
        "order of operations": "The order of operations is remembered by PEMDAS: Parentheses, Exponents, Multiplication and Division (left to right), then Addition and Subtraction (left to right). Example: 2 + 3 × 4 = 2 + 12 = 14 (not 20).",
        "derivative": "A derivative measures the rate of change of a function. If f(x) = xⁿ, then f'(x) = nxⁿ⁻¹ (the power rule). For example, if f(x) = x³, then f'(x) = 3x². Derivatives are used to find slopes, maxima, minima, and rates of change.",
        "area of a circle": "The area of a circle is A = πr², where r is the radius. The circumference is C = 2πr. For example, a circle with radius 5 has area = π(5²) = 25π ≈ 78.54 square units.",
        "prime number": "A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself. Examples: 2, 3, 5, 7, 11, 13. The number 2 is the only even prime. Numbers with more than two factors are called composite numbers.",
        "slope": "The slope of a line measures its steepness and is calculated as m = (y₂ - y₁) / (x₂ - x₁), the rise over run. A positive slope goes up left to right, negative goes down. The slope-intercept form of a line is y = mx + b, where b is the y-intercept.",
        "integral": "An integral is the reverse of differentiation and represents the area under a curve. The indefinite integral of xⁿ is xⁿ⁺¹/(n+1) + C. The definite integral ∫[a to b] f(x)dx gives the exact area between the curve and the x-axis from a to b.",
        "mean median mode": "Mean is the average (sum divided by count). Median is the middle value when data is sorted. Mode is the most frequently occurring value. For example, in {2, 3, 3, 5, 7}: mean = 4, median = 3, mode = 3.",
        "fraction": "To simplify a fraction, divide both numerator and denominator by their Greatest Common Factor (GCF). For example, 12/18: the GCF of 12 and 18 is 6, so 12/18 = 2/3. To add fractions, find a common denominator first.",
        "matrix": "A matrix is a rectangular array of numbers arranged in rows and columns. Matrices are used to solve systems of equations, represent transformations, and in data science. Matrix multiplication requires the number of columns in the first matrix to equal the rows in the second.",
        "logarithm": "A logarithm answers: 'to what power must we raise the base to get a number?' log_b(x) = y means b^y = x. For example, log₁₀(100) = 2 because 10² = 100. The natural log (ln) uses base e ≈ 2.718.",
        "calculus": "The Fundamental Theorem of Calculus links differentiation and integration. It states that if F is an antiderivative of f on [a,b], then ∫[a to b] f(x)dx = F(b) - F(a). This means integration and differentiation are inverse operations.",
        "volume of a sphere": "The volume of a sphere is V = (4/3)πr³, where r is the radius. The surface area is A = 4πr². For example, a sphere with radius 3: V = (4/3)π(27) = 36π ≈ 113.1 cubic units.",
        "function": "A function is a relation where each input has exactly one output. Written as f(x), it maps every value of x to exactly one value of y. Functions can be linear (f(x) = mx+b), quadratic (f(x) = ax²+bx+c), exponential, logarithmic, and more.",
        "permutations combinations": "Permutations count ordered arrangements: P(n,r) = n!/(n-r)!. Combinations count unordered selections: C(n,r) = n!/[r!(n-r)!]. Example: Choosing 3 from 5 people — permutations = 60, combinations = 10.",
        "pi": "Pi (π) is the ratio of a circle's circumference to its diameter, approximately 3.14159. It is an irrational and transcendental number, meaning it cannot be expressed as a simple fraction and its decimal expansion never repeats. It appears in many formulas involving circles and waves.",
        "probability": "Probability measures the likelihood of an event, expressed as P(event) = favorable outcomes / total outcomes, ranging from 0 (impossible) to 1 (certain). For example, rolling a 3 on a die: P = 1/6 ≈ 0.167 or 16.7%.",
        "standard deviation": "Standard deviation measures how spread out data is from the mean. A low standard deviation means data points are close to the mean; a high one means they are spread out. Formula: σ = √(Σ(xᵢ - μ)² / N) for a population.",
        "exponent": "An exponent indicates how many times a base is multiplied by itself. For example, 2⁴ = 2×2×2×2 = 16. Key rules: xᵃ × xᵇ = xᵃ⁺ᵇ, (xᵃ)ᵇ = xᵃᵇ, x⁰ = 1, x⁻ᵃ = 1/xᵃ.",
        "default": "Great math question! I cover middle school through college: arithmetic, pre-algebra, algebra, geometry, statistics, precalculus, and calculus. Send a problem or concept and I will walk through it at the level you need.",
    },
    "english": {
        "metaphor": "A metaphor is a figure of speech that directly describes something as if it were something else, without using 'like' or 'as.' Example: 'Life is a journey.' It creates a direct comparison to highlight a quality. Unlike a simile, it does not use comparative words.",
        "simile": "A simile compares two things using 'like' or 'as.' Example: 'She runs like the wind.' A metaphor makes the same comparison directly: 'She is the wind.' Both are figures of speech used to create vivid imagery and deepen meaning in writing.",
        "thesis statement": "A thesis statement is a sentence (usually at the end of the introduction) that states the main argument or claim of an essay. A strong thesis is specific, arguable, and gives the reader a roadmap for the paper. Example: 'Social media harms teenage mental health by promoting unrealistic beauty standards.'",
        "topic sentence": "A topic sentence is the first sentence of a body paragraph that states the main idea of that paragraph. It connects back to the thesis and tells the reader what the paragraph will discuss. Every supporting detail in the paragraph should relate to the topic sentence.",
        "foreshadowing": "Foreshadowing is a literary device where the author gives subtle hints about what will happen later in the story. It builds suspense and prepares the reader. Example: In 'Romeo and Juliet,' Romeo says 'my life were better ended by their hate than death prorogued' — hinting at his death.",
        "active passive voice": "In active voice, the subject performs the action: 'The dog chased the cat.' In passive voice, the subject receives the action: 'The cat was chased by the dog.' Active voice is generally preferred in writing because it is clearer and more direct.",
        "irony": "Irony occurs when there is a contrast between expectation and reality. Verbal irony: saying the opposite of what you mean (sarcasm). Situational irony: what happens is opposite to what is expected. Dramatic irony: the audience knows something the characters don't.",
        "allegory": "An allegory is a story in which characters and events are symbols representing deeper moral, political, or spiritual meanings. Example: George Orwell's 'Animal Farm' is an allegory for the Russian Revolution, where the pigs represent the Soviet leadership.",
        "conclusion paragraph": "A strong conclusion restates the thesis (in new words), summarizes key points, and ends with a broader insight or call to action. It should not introduce new information. Think of it as answering the question: 'So what? Why does this matter?'",
        "characterization": "Characterization is how an author develops and reveals a character's personality. Direct characterization explicitly states traits ('She was kind'). Indirect characterization shows traits through actions, dialogue, thoughts, and reactions of others — often remembered by the acronym STEAL.",
        "protagonist antagonist": "The protagonist is the main character the story follows, often facing a central conflict. The antagonist is the opposing force — a character, group, nature, or internal struggle that creates obstacles for the protagonist. Example: Harry Potter is the protagonist; Voldemort is the antagonist.",
        "alliteration": "Alliteration is the repetition of the same initial consonant sound in nearby words. Example: 'Peter Piper picked a peck of pickled peppers.' It is used in poetry, literature, and advertising to create rhythm and make phrases memorable.",
        "personification": "Personification gives human qualities to non-human things or abstract ideas. Example: 'The wind howled through the night' gives the wind a human action. It makes descriptions more vivid and relatable by connecting the subject to human experience.",
        "connotation denotation": "Denotation is the literal, dictionary definition of a word. Connotation is the emotional or cultural association. For example, 'home' and 'house' have similar denotations, but 'home' connotes warmth and family, while 'house' is more neutral.",
        "symbolism": "Symbolism is when an object, person, place, or event represents something beyond its literal meaning. Example: A dove symbolizes peace; a green light in 'The Great Gatsby' symbolizes Gatsby's dreams and the American Dream.",
        "plot structure": "Plot structure typically follows Freytag's Pyramid: Exposition (background), Rising Action (conflict builds), Climax (turning point), Falling Action (consequences), and Resolution (conclusion). This structure gives stories a satisfying arc.",
        "fiction nonfiction": "Fiction is imaginative writing with invented characters, events, and settings (novels, short stories, plays). Nonfiction is based on real events, people, and facts (biographies, essays, journalism, textbooks). Both can use literary devices and storytelling techniques.",
        "imagery": "Imagery uses descriptive language to appeal to the five senses — sight, sound, smell, taste, and touch. It helps readers visualize and feel the scene. Example: 'The sharp scent of pine filled the cold mountain air' creates a vivid sensory experience.",
        "oxymoron": "An oxymoron combines two contradictory terms to create a new, complex meaning. Examples: 'bittersweet,' 'deafening silence,' 'living death.' Shakespeare used them frequently — 'O heavy lightness! Serious vanity!' in Romeo and Juliet.",
        "haiku": "A haiku is a traditional Japanese poem with three lines following a 5-7-5 syllable pattern. It typically captures a moment in nature and juxtaposes two ideas. Example: 'An old silent pond / A frog jumps into the pond / Splash! Silence again.' — Matsuo Bashō.",
        "default": "Great English question! I cover middle school through college: grammar, reading strategies, literary devices, poetry, essays, and analysis. Name a text, device, or assignment prompt for a targeted answer.",
    },
    "art": {
        "color theory": "Color theory explains how colors interact. The color wheel includes primary colors (red, yellow, blue), secondary colors (green, orange, purple), and tertiary colors. Complementary colors are opposite on the wheel (red/green). Warm colors advance; cool colors recede. Value (lightness/darkness) affects mood.",
        "leonardo da vinci": "Leonardo da Vinci (1452–1519) was an Italian Renaissance polymath — painter, sculptor, architect, scientist, and inventor. His masterworks include the Mona Lisa and The Last Supper. He pioneered sfumato (subtle tonal blending) and studied anatomy to achieve unprecedented realism.",
        "impressionism": "Impressionism was a 19th-century art movement originating in France. Artists like Monet, Renoir, and Degas painted scenes with loose brushstrokes and an emphasis on capturing light and atmosphere rather than precise detail. Monet's 'Impression, Sunrise' gave the movement its name.",
        "pablo picasso": "Pablo Picasso (1881–1973) was a Spanish painter and co-founder of Cubism. His most famous works include 'Guernica' (a powerful anti-war statement) and 'Les Demoiselles d'Avignon.' He went through distinct periods — Blue, Rose, African, Cubist — each with different styles.",
        "abstract art": "Abstract art uses shapes, colors, forms, and gestural marks to create a composition that may have little or no visual reference to the real world. Key figures include Wassily Kandinsky, Jackson Pollock, and Mark Rothko. The focus is on emotion, concept, and form rather than representation.",
        "renaissance": "The Renaissance (14th–17th century) was a cultural rebirth in Europe that revived classical Greek and Roman ideas. In art, it brought linear perspective, anatomical accuracy, and humanist themes. Key artists: Leonardo da Vinci, Michelangelo, Raphael, and Botticelli.",
        "composition": "Composition in art refers to the arrangement of visual elements within a work. Principles include the rule of thirds, balance (symmetrical vs. asymmetrical), focal point, leading lines, and negative space. Good composition guides the viewer's eye and creates harmony.",
        "perspective": "Perspective in drawing creates the illusion of three-dimensional depth on a flat surface. One-point perspective uses a single vanishing point (e.g., a road going to the horizon). Two-point perspective uses two vanishing points. It was developed during the Renaissance by Brunelleschi.",
        "chiaroscuro": "Chiaroscuro (Italian for 'light-dark') is a technique using strong contrasts between light and shadow to give the illusion of three-dimensionality. Mastered by Caravaggio and Rembrandt, it creates dramatic, sculptural effects and emotional intensity.",
        "vincent van gogh": "Vincent van Gogh (1853–1890) was a Dutch Post-Impressionist painter whose work had a major influence on 20th-century art. Famous works include 'The Starry Night' and 'Sunflowers.' He used bold colors and expressive, swirling brushstrokes. He sold only one painting during his lifetime.",
        "surrealism": "Surrealism was a 20th-century art and literary movement that explored the unconscious mind and dreamlike imagery. Founded by André Breton, key artists include Salvador Dalí ('The Persistence of Memory'), René Magritte, and Max Ernst. It drew heavily on Freudian psychology.",
        "primary colors": "The traditional primary colors in art are red, yellow, and blue — they cannot be created by mixing other colors but can be combined to make all others. In light (RGB), primaries are red, green, and blue. Mixing red+yellow=orange, yellow+blue=green, red+blue=purple.",
        "pointillism": "Pointillism is a technique developed by Georges Seurat in the 1880s where small, distinct dots of pure color are applied to a canvas. When viewed from a distance, the dots blend optically to create the impression of a range of colors. 'A Sunday on La Grande Jatte' is his masterpiece.",
        "michelangelo": "Michelangelo (1475–1564) was an Italian Renaissance sculptor, painter, and architect. He created the marble statue of David, the Pietà, and painted the ceiling of the Sistine Chapel. He considered himself primarily a sculptor; the Sistine Chapel was originally a reluctant commission.",
        "pop art": "Pop Art emerged in the 1950s–60s in Britain and the U.S., drawing inspiration from popular culture, advertising, comic books, and consumer goods. Key artists: Andy Warhol (Campbell's Soup Cans), Roy Lichtenstein (comic-style paintings), and Jasper Johns (flags and targets).",
        "golden ratio": "The golden ratio (approximately 1.618, denoted φ) is a mathematical proportion found in nature and considered aesthetically pleasing. It appears in the Parthenon, Da Vinci's works, and nautilus shells. Designers and artists use it to create balanced, harmonious compositions.",
        "cubism": "Cubism was a revolutionary 20th-century art movement developed by Picasso and Braque. It broke objects into geometric fragments and depicted them from multiple viewpoints simultaneously. Analytic Cubism used monochromatic tones; Synthetic Cubism incorporated collage elements.",
        "frida kahlo": "Frida Kahlo (1907–1954) was a Mexican painter known for her deeply personal self-portraits that blended realism, symbolism, and surrealism. Her work explored identity, post-colonialism, gender, and physical and psychological pain. She is a feminist icon and symbol of Mexican national identity.",
        "negative space": "Negative space is the area surrounding the subject (positive space) in an artwork. Artists like M.C. Escher used negative space creatively to create optical illusions. Understanding negative space helps artists achieve better balance and composition.",
        "expressionism": "Expressionism was an early 20th-century movement, particularly in Germany, that prioritized the expression of emotional experience over physical reality. Artists like Edvard Munch ('The Scream') and Ernst Ludwig Kirchner used distorted forms and vivid colors to convey anxiety and alienation.",
        "baroque": "The Baroque period (1600–1750) in art was characterized by dramatic use of light and shadow (chiaroscuro), intense emotional expression, and grandiose scale. Key artists include Caravaggio, Rembrandt, and Peter Paul Rubens. It was often used by the Catholic Church to inspire awe.",
        "default": "Great art question! I cover middle school through college: studio basics, design principles, and art history from ancient to contemporary. Ask about a movement, artwork, or technique.",
    },
    "politics": {
        "democracy": "Democracy is a system of government where power is vested in the people, who exercise it directly or through elected representatives. Direct democracy (citizens vote on laws directly) vs. representative democracy (citizens elect officials to vote on their behalf). The U.S. is a constitutional representative democracy.",
        "republic democracy": "A democracy is governed by majority rule. A republic is a form of government where the country is considered a public matter, with elected representatives and a constitution that limits government power. The U.S. is both — a democratic republic. Pure democracy risks 'tyranny of the majority.'",
        "three branches": "The U.S. government has three branches: (1) Legislative — Congress (Senate + House of Representatives), which makes laws. (2) Executive — the President and Cabinet, which enforces laws. (3) Judicial — the Supreme Court and federal courts, which interpret laws.",
        "federalism": "Federalism is the division of power between a central (federal) government and state governments. The U.S. Constitution grants certain powers to the federal government (national defense, foreign policy) while reserving others for states (education, local laws).",
        "electoral college": "The Electoral College is the system used to elect the U.S. President. Each state has electors equal to its Congressional representation. A candidate needs 270 of 538 electoral votes to win. Critics argue it can allow a candidate to win without the popular vote majority.",
        "political party": "A political party is an organized group that shares common political ideologies and works to elect candidates to government office. The U.S. has a two-party system dominated by Democrats (center-left) and Republicans (center-right), though third parties also exist.",
        "capitalism": "Capitalism is an economic and political system where private individuals or corporations own the means of production and operate for profit. Prices and production are determined by free market competition. The U.S., U.K., and most Western nations operate as mixed capitalist economies.",
        "socialism": "Socialism is an economic system where the means of production are owned collectively or by the state, and wealth is distributed more equally. It exists on a spectrum — from democratic socialism (Scandinavia) to authoritarian socialism. Key ideas: public ownership, social welfare, and wealth redistribution.",
        "supreme court": "The Supreme Court is the highest federal court in the U.S., consisting of 9 justices appointed for life by the President and confirmed by the Senate. It has the power of judicial review — it can declare laws unconstitutional. Landmark cases include Brown v. Board of Education and Roe v. Wade.",
        "filibuster": "A filibuster is a parliamentary tactic in the U.S. Senate where a senator delays or prevents a vote on a bill by speaking for an extended time. The record is held by Strom Thurmond (24 hours, 18 minutes). A filibuster can be ended by a cloture vote requiring 60 senators.",
        "separation of powers": "Separation of powers divides government authority among three independent branches (legislative, executive, judicial) to prevent any one branch from gaining too much power. Combined with checks and balances, each branch has ways to limit the others.",
        "bill of rights": "The Bill of Rights is the first 10 amendments to the U.S. Constitution, ratified in 1791. They protect fundamental rights including freedom of speech, religion, and press (1st), the right to bear arms (2nd), protection from unreasonable searches (4th), and due process (5th).",
        "gerrymandering": "Gerrymandering is the manipulation of electoral district boundaries to favor a particular political party or group. It can dilute minority votes by packing them into one district or cracking them across many. The term comes from Governor Elbridge Gerry, who approved a salamander-shaped district in 1812.",
        "liberal conservative": "Liberalism generally supports government intervention in the economy to promote equality, individual rights, and social programs. Conservatism generally favors limited government, free markets, traditional values, and a strong national defense. These are broad labels with much variation within each.",
        "veto": "A veto is the power of the President to reject a bill passed by Congress. Congress can override a veto with a two-thirds majority vote in both the House and Senate. The pocket veto occurs when the President neither signs nor vetoes a bill within 10 days while Congress is adjourned.",
        "nato": "NATO (North Atlantic Treaty Organization), founded in 1949, is a military alliance of 32 countries in North America and Europe. Its core principle is collective defense — an attack on one member is considered an attack on all (Article 5). It was formed as a counterbalance to the Soviet Union.",
        "united nations": "The United Nations (UN), founded in 1945 after World War II, is an international organization with 193 member states aimed at maintaining international peace, security, and cooperation. Its main bodies include the Security Council (5 permanent members with veto power) and the General Assembly.",
        "propaganda": "Propaganda is the dissemination of information, especially biased or misleading information, to promote a political cause or ideology. Techniques include bandwagon, fear appeals, glittering generalities, and card stacking. It has been used by all political systems throughout history.",
        "checks balances": "Checks and balances is a system where each branch of government has powers that limit the other branches. Examples: Congress passes laws but the President can veto; the President appoints judges but the Senate confirms; courts can strike down laws as unconstitutional.",
        "amendment": "A Constitutional amendment is a formal change to the U.S. Constitution. There are 27 amendments. The process requires two-thirds approval from both houses of Congress and ratification by three-fourths of states. Key amendments: 13th (abolished slavery), 19th (women's suffrage), 26th (voting age 18).",
        "default": "Great politics question! I cover middle school civics through college: institutions, elections, rights, ideologies, and international relations. Ask about a concept, case, or current-events link to coursework.",
    },
    "biology": {
        "cell": "A cell is the basic structural and functional unit of all living organisms. Prokaryotic cells (bacteria) lack a nucleus; eukaryotic cells (plants, animals, fungi) have a membrane-bound nucleus. Key organelles include the nucleus (DNA), mitochondria (energy), ribosomes (protein synthesis), and in plants, chloroplasts.",
        "dna": "DNA (Deoxyribonucleic Acid) is the molecule that carries genetic information in all living organisms and many viruses. It is a double helix made of nucleotides containing the bases adenine (A), thymine (T), guanine (G), and cytosine (C). A pairs with T; G pairs with C.",
        "mitosis meiosis": "Mitosis produces two identical diploid daughter cells for growth and repair. It has 4 phases: Prophase, Metaphase, Anaphase, Telophase (PMAT). Meiosis produces four genetically diverse haploid cells (sperm or eggs) for sexual reproduction, and involves two rounds of division.",
        "natural selection": "Natural selection is Darwin's mechanism of evolution: organisms with traits better suited to their environment survive and reproduce more successfully, passing those traits to offspring. Over generations, this leads to adaptation. It requires variation, heritability, and differential reproductive success.",
        "photosynthesis": "Photosynthesis is the process by which plants, algae, and some bacteria convert light energy into chemical energy. The equation is: 6CO₂ + 6H₂O + light energy → C₆H₁₂O₆ + 6O₂. It occurs in chloroplasts, using chlorophyll to absorb sunlight.",
        "cell membrane": "The cell membrane (plasma membrane) is a selectively permeable phospholipid bilayer that surrounds all cells. It controls what enters and exits the cell. The fluid mosaic model describes it as a flexible structure with embedded proteins that act as channels, receptors, and enzymes.",
        "ecosystem": "An ecosystem is a community of living organisms (biotic factors) interacting with their non-living environment (abiotic factors — sunlight, water, soil, temperature). Ecosystems cycle nutrients and energy flows through food webs. Examples: rainforests, coral reefs, deserts, and ponds.",
        "chromosome": "Chromosomes are thread-like structures of DNA and protein found in the nucleus of eukaryotic cells. Humans have 46 chromosomes (23 pairs). Chromosomes carry genes — sections of DNA that encode for proteins. Sex chromosomes (XX = female, XY = male) determine biological sex.",
        "prokaryotes eukaryotes": "Prokaryotes (bacteria and archaea) are single-celled organisms without a membrane-bound nucleus — their DNA floats in the cytoplasm. Eukaryotes (animals, plants, fungi, protists) have a true nucleus and membrane-bound organelles. Eukaryotic cells are generally larger and more complex.",
        "evolution": "Evolution is the change in heritable characteristics of populations over successive generations. Charles Darwin proposed natural selection as its primary mechanism. Evidence comes from the fossil record, comparative anatomy, DNA similarities, and direct observation of adaptation.",
        "cellular respiration": "Cellular respiration is how cells convert glucose into ATP (usable energy). The equation is: C₆H₁₂O₆ + 6O₂ → 6CO₂ + 6H₂O + ATP. It occurs in three stages: Glycolysis (cytoplasm), Krebs Cycle, and the Electron Transport Chain (both in mitochondria).",
        "gene": "A gene is a segment of DNA that encodes the instructions for making a specific protein or RNA molecule. Genes are the units of heredity. The human genome contains approximately 20,000–25,000 genes. Genotype (genetic makeup) interacts with environment to produce phenotype (observable traits).",
        "rna dna": "DNA (double-stranded) stores genetic information in the nucleus. RNA (single-stranded) carries out DNA's instructions. mRNA carries the genetic code from nucleus to ribosomes; tRNA brings amino acids; rRNA forms ribosomes. RNA uses uracil (U) instead of thymine (T).",
        "osmosis": "Osmosis is the movement of water molecules across a selectively permeable membrane from an area of higher water concentration (lower solute) to lower water concentration (higher solute). It continues until equilibrium. This is critical for cell function — cells shrivel (crenation) or burst (lysis) under extreme tonicity mismatches.",
        "mutation": "A mutation is a change in the DNA sequence. Point mutations change one base; insertions/deletions shift the reading frame. Mutations can be caused by radiation, chemicals, or errors in DNA replication. Some mutations are neutral, some harmful, and rarely some are beneficial (driving evolution).",
        "nervous system": "The nervous system coordinates the body's responses to internal and external stimuli. The Central Nervous System (CNS) — brain and spinal cord — processes information. The Peripheral Nervous System (PNS) transmits signals between the CNS and the rest of the body. Neurons are the basic signaling units.",
        "virus bacteria": "Bacteria are single-celled prokaryotic organisms that can reproduce independently and live in virtually any environment. Viruses are not considered alive — they are non-cellular packages of DNA or RNA that can only replicate inside a host cell. Antibiotics treat bacterial infections but do not work on viruses.",
        "enzyme": "Enzymes are biological catalysts — proteins that speed up chemical reactions without being consumed. Each enzyme has an active site specific to its substrate (lock-and-key model). Temperature and pH affect enzyme activity. Enzymes are critical for digestion, DNA replication, and metabolism.",
        "atp": "ATP (Adenosine Triphosphate) is the primary energy currency of the cell. When its terminal phosphate bond is broken (ATP → ADP + Pᵢ), energy is released to power cellular work including muscle contraction, active transport, and biosynthesis. It is produced by cellular respiration.",
        "crispr": "CRISPR-Cas9 is a revolutionary gene-editing technology that allows scientists to precisely cut, edit, or replace segments of DNA. It works like molecular scissors guided by RNA. CRISPR has applications in treating genetic diseases, developing crops, and basic research. It raises significant ethical debates.",
        "default": "Great biology question! I cover middle school life science through college: cells, genetics, body systems, evolution, and ecology. Ask about a process, structure, or vocabulary term.",
    },
    "physics": {
        "newton": "Newton's First Law (Law of Inertia): An object at rest stays at rest, and an object in motion stays in motion with the same speed and direction, unless acted upon by an unbalanced external force. Newton's Second Law: F = ma. Third Law: For every action, there is an equal and opposite reaction.",
        "speed of light": "The speed of light in a vacuum is approximately 299,792,458 meters per second (about 3×10⁸ m/s), denoted as 'c'. According to Einstein's Special Relativity, nothing with mass can travel at or faster than c. Light from the Sun takes about 8 minutes to reach Earth.",
        "gravity": "Gravity is the attractive force between objects with mass. Newton's Law of Universal Gravitation: F = Gm₁m₂/r², where G is the gravitational constant. Einstein's General Relativity describes gravity as the curvature of spacetime caused by mass and energy.",
        "mass weight": "Mass is the amount of matter in an object (measured in kg) and is constant everywhere. Weight is the gravitational force acting on an object: W = mg, where g is gravitational acceleration (≈9.8 m/s² on Earth). On the Moon (g ≈ 1.6 m/s²), you weigh less but your mass stays the same.",
        "energy": "Energy is the capacity to do work. Forms include kinetic (motion), potential (stored), thermal (heat), chemical, nuclear, and electromagnetic. The SI unit is the Joule (J). Energy can be converted between forms but cannot be created or destroyed — the Law of Conservation of Energy.",
        "conservation of energy": "The Law of Conservation of Energy states that energy cannot be created or destroyed, only converted from one form to another. Total energy in a closed system remains constant. For example, a falling object converts potential energy (PE = mgh) to kinetic energy (KE = ½mv²).",
        "wave": "A wave is a disturbance that transfers energy through matter or space. Transverse waves (light, electromagnetic) vibrate perpendicular to direction of travel. Longitudinal waves (sound) vibrate parallel to direction. Key properties: wavelength (λ), frequency (f), amplitude, and speed (v = fλ).",
        "kinetic potential energy": "Kinetic energy is the energy of motion: KE = ½mv². Potential energy is stored energy due to position or condition. Gravitational PE = mgh. Elastic PE = ½kx². As an object falls, PE converts to KE; at the bottom, all energy is kinetic. Total mechanical energy = KE + PE (constant without friction).",
        "electromagnetism": "Electromagnetism is the interaction between electric charges and currents. Maxwell's equations unify electricity and magnetism, showing they are aspects of one force. Electromagnetic waves (light, radio, X-rays) travel at the speed of light and require no medium.",
        "quantum mechanics": "Quantum mechanics describes the behavior of matter and energy at the atomic and subatomic scale. Key principles: wave-particle duality, quantized energy levels, Heisenberg's Uncertainty Principle, and Schrödinger's wave equation. It explains phenomena classical physics cannot — atomic spectra, tunneling, and superconductivity.",
        "relativity": "Einstein's Special Relativity (1905) states that the laws of physics are the same in all inertial frames and the speed of light is constant. It leads to time dilation and length contraction. General Relativity (1915) describes gravity as the curvature of spacetime by mass and energy. E = mc².",
        "black hole": "A black hole is a region of spacetime where gravity is so extreme that nothing — not even light — can escape from it. They form when massive stars collapse under their own gravity. The boundary from which nothing can escape is called the event horizon. Supermassive black holes exist at the center of most galaxies.",
        "friction": "Friction is a force that opposes the relative motion between two surfaces in contact. Static friction prevents stationary objects from moving; kinetic friction acts on moving objects. Friction force = μN, where μ is the coefficient of friction and N is the normal force.",
        "thermodynamics": "The Laws of Thermodynamics: 0th — if A=B and B=C in thermal equilibrium, then A=C. 1st — energy is conserved (ΔU = Q - W). 2nd — entropy of an isolated system always increases; heat flows from hot to cold. 3rd — absolute zero (0 K) is unattainable.",
        "momentum": "Momentum is the product of an object's mass and velocity: p = mv (vector quantity, measured in kg·m/s). The Law of Conservation of Momentum states that in a closed system, total momentum is conserved. In collisions: elastic (KE conserved) vs. inelastic (KE lost).",
        "ohm's law": "Ohm's Law states that the current through a conductor is directly proportional to the voltage across it: V = IR, where V is voltage (volts), I is current (amperes), and R is resistance (ohms). It is foundational in circuit analysis.",
        "nuclear fission": "Nuclear fission is when a heavy atomic nucleus (like uranium-235 or plutonium-239) splits into smaller nuclei, releasing enormous amounts of energy. This is the principle behind nuclear power plants and atomic bombs. The energy comes from the mass difference (E=mc²).",
        "speed velocity": "Speed is a scalar quantity — it has magnitude only (e.g., 60 km/h). Velocity is a vector quantity — it has both magnitude and direction (e.g., 60 km/h north). Acceleration is the rate of change of velocity: a = Δv/Δt.",
        "photon": "A photon is a massless quantum (particle) of light and all electromagnetic radiation. It has energy E = hf, where h is Planck's constant and f is frequency. Photons exhibit wave-particle duality — they behave as both waves and particles depending on how they are measured.",
        "dark matter": "Dark matter is a hypothetical type of matter that does not emit, absorb, or reflect light, making it undetectable by electromagnetic radiation. It is estimated to make up about 27% of the universe's total mass-energy content. Its existence is inferred from gravitational effects on visible matter and galaxies.",
        "entropy": "Entropy is a measure of disorder or randomness in a system, central to the Second Law of Thermodynamics. In any spontaneous process, total entropy increases. This explains why heat flows from hot to cold, why ice melts at room temperature, and why ordered systems naturally tend toward disorder.",
        "doppler effect": "The Doppler effect is the change in frequency of a wave (sound or light) as the source and observer move relative to each other. When they approach, frequency increases (higher pitch); when moving apart, frequency decreases (lower pitch). It explains why a siren sounds different as it passes by.",
        "heisenberg": "Heisenberg's Uncertainty Principle states that it is impossible to precisely and simultaneously know both the position and momentum of a particle: Δx·Δp ≥ ℏ/2. The more precisely position is known, the less precisely momentum can be known, and vice versa. It is a fundamental property of quantum systems.",
        "string theory": "String theory is a theoretical framework in physics that proposes that all fundamental particles are not point-like but are tiny, one-dimensional vibrating strings of energy. Different vibration modes correspond to different particles. It aims to unify quantum mechanics and general relativity but remains unproven.",
        "boiling point of water": "At standard sea-level air pressure (about 101.3 kPa or 1 atm), pure water boils at 100 degrees Celsius (212 degrees Fahrenheit). Boiling is a phase change from liquid to gas: molecules gain enough thermal energy to escape the liquid as vapor. On a tall mountain, lower air pressure lowers the boiling point, so water can boil near 70 degrees C on Everest — that is why pasta takes longer to cook at altitude.",
        "melting point of ice": "Pure ice melts at 0 degrees Celsius (32 degrees Fahrenheit) at standard pressure. Melting is solid to liquid; freezing is the reverse at the same temperature for pure water. Adding salt lowers the freezing point (colligative property), which is why roads are salted before ice storms.",
        "density": "Density is mass per unit volume, rho = m/V, usually in kg/m^3 or g/cm^3. Less dense objects float on denser fluids (Archimedes principle). Water has about 1000 kg/m^3; air is roughly 1.2 kg/m^3 at sea level.",
        "newton second law": "Newton's second law states F = m a: net force on an object equals its mass times acceleration (vector equation). In middle school problems, forces are often along a line; sum forces in each direction separately.",
        "default": "Great physics question! I cover middle school through college: mechanics, heat and phase changes, waves, electricity, and modern topics. Ask about a specific law, problem, or phenomenon and I will answer at the level you need.",
    },
    "chemistry": {
        "periodic table": "The periodic table groups elements by atomic number (protons) and repeating chemical properties. Rows are periods; columns are groups with similar valence electrons. Metals are usually left and center, nonmetals to the right, metalloids along the staircase. It is the main map chemists use to predict bonding and reactivity.",
        "atom": "An atom has a dense nucleus (protons and neutrons) surrounded by electrons in orbitals. The atomic number Z equals the number of protons and defines the element. In neutral atoms, electrons equal protons. Ions form when electrons are gained or lost.",
        "molecule": "A molecule is two or more atoms bonded together as a distinct unit (e.g., O2, H2O, CO2). Molecular compounds share electrons (covalent bonds). Ionic compounds like NaCl are lattices of ions, not separate molecules in the solid state.",
        "h2o": "H2O is water’s chemical formula: two hydrogen atoms covalently bonded to one oxygen in a bent shape. It is a polar molecule, an excellent solvent for many ions and polar compounds, and central to acid–base chemistry (H+ and OH- in aqueous solution) and hydrogen bonding.",
        "element compound": "An element is one kind of atom (e.g., carbon). A compound contains two or more elements chemically bonded in fixed ratio (e.g., H2O). Mixtures can vary in composition and can often be separated physically.",
        "ionic covalent bond": "Ionic bonds form when electrons transfer from metal to nonmetal, creating oppositely charged ions that attract (Na+ and Cl-). Covalent bonds share electrons between nonmetals. Bond type depends on electronegativity difference: large difference tends toward ionic; small difference tends toward covalent.",
        "acid base": "Arrhenius acids increase H+ in water; bases increase OH-. Bronsted acids donate protons; bases accept them. The pH scale is pH = -log10[H+]; pH 7 is neutral at 25 C, below 7 acidic, above 7 basic. Neutralization often produces water and a salt.",
        "ph scale": "pH measures acidity from 0 (very acidic) to 14 (very basic), with 7 neutral for pure water at 25 C. Each step of 1 on pH means a tenfold change in hydrogen ion concentration. Indicators and pH meters are common measurement tools in lab and field.",
        "mole": "The mole is SI amount of substance: 1 mol = 6.022 x 10^23 particles (Avogadro constant). Molar mass (g/mol) converts between grams and moles. Stoichiometry uses balanced equations to relate moles of reactants and products.",
        "balancing chemical equations": "Balance atoms on both sides by adjusting coefficients (never change subscripts — that would change the substance). Start with elements that appear in only one compound on each side, then finish with hydrogen and oxygen last in combustion reactions.",
        "states of matter": "Common states: solid (fixed shape, vibrates in place), liquid (fixed volume, flows), gas (fills container, widely spaced particles), plasma (ionized gas, conducts). Phase changes include melting, freezing, vaporization, condensation, sublimation, and deposition — each involves latent heat at constant temperature for pure substances at fixed pressure.",
        "catalyst": "A catalyst speeds a reaction without being consumed overall by providing a lower-energy pathway. Enzymes are biological catalysts. Catalysts appear unchanged at the end of a cycle but often participate in intermediate steps.",
        "oxidation reduction": "Redox involves electron transfer: oxidation is loss of electrons (increase in oxidation number), reduction is gain (decrease in oxidation number). The species oxidized is the reducing agent; the species reduced is the oxidizing agent. Balancing redox in acid or base uses half-reactions.",
        "organic chemistry basics": "Organic chemistry studies carbon compounds: hydrocarbons (alkanes, alkenes, alkynes), functional groups (alcohols, carboxylic acids, esters, amines), and reaction types (substitution, addition). Carbon forms four bonds and chains, enabling enormous molecular diversity.",
        "gas laws": "Ideal gas law: PV = nRT relates pressure, volume, moles, and temperature. Boyle law (n,T fixed): P inversely proportional to V. Charles law (n,P fixed): V proportional to T. Avogadro law: V proportional to n at same P,T. Real gases deviate at high pressure and low temperature.",
        "solution solubility": "A solution is a homogeneous mixture of solute in solvent. Solubility is the maximum solute that dissolves at a given temperature. Like dissolves like: polar solvents dissolve many ionic and polar solutes; nonpolar solvents dissolve nonpolar solutes.",
        "isotope": "Isotopes of an element have the same number of protons but different numbers of neutrons (different mass number). They share chemistry but differ in mass and nuclear stability. Some isotopes are radioactive.",
        "electronegativity": "Electronegativity is an atom tendency to attract shared electrons in a bond. Fluorine is highest on common scales. It helps predict bond polarity and whether a bond is ionic, polar covalent, or nonpolar covalent.",
        "enthalpy": "Enthalpy H is thermodynamic energy-like quantity at constant pressure; delta H is heat flow for many lab reactions. Exothermic releases heat (negative delta H); endothermic absorbs heat (positive delta H). Hess law sums reaction steps.",
        "lewis structure": "Lewis structures show valence electrons as dots or lines (bond pairs). They predict bonding and lone pairs and formal charge. Octet rule guides many main-group structures, with exceptions for expanded octets in heavier elements.",
        "stoichiometry limiting reagent": "From a balanced equation, convert masses to moles, then use mole ratios to find theoretical yield. The limiting reagent is consumed first and caps product amount. Percent yield = (actual yield / theoretical yield) x 100%.",
        "chemical nomenclature": "Chemists use systematic IUPAC names and common (trivial) names. IUPAC rules encode structure: parent chain, numbering, prefixes for substituents, and suffixes for the principal functional group. Common names (water, ammonia, acetone) remain widely spoken; exams often require IUPAC for synthesis and mechanism problems.",
        "iupac name": "IUPAC nomenclature builds a unique name from structure (longest chain, lowest locants, alphabetical substituent order, priority of functional groups). For example CH3CH2OH is ethanol (common) and systematically ethanol as well; branched cases add locants like 2-methylbutane.",
        "common name vs systematic name": "A common name is what people say in the lab (acetic acid); the systematic (IUPAC) name maps cleanly to structure (ethanoic acid). Many ions and acids have both; always follow your instructor's naming convention on assessments.",
        "default": "Strong chemistry question! I support middle school through college: atoms and bonding, reactions, acids and bases, gases, solutions, and introductory organic chemistry. Name a concept or paste a homework prompt for a focused explanation.",
    },
    "technology": {
        "computer basics": "A computer processes data using hardware (CPU, memory, storage, input/output) and software (programs). The CPU executes instructions from memory; RAM is fast volatile workspace; storage (SSD/HDD) keeps data when powered off. Operating systems manage hardware and programs.",
        "cpu": "The CPU (central processing unit) executes machine instructions. Cores run streams of instructions; clock speed (GHz) is one performance factor along with architecture, cache, and instruction count per task. Multithreading lets cores work on several tasks when programs support it.",
        "ram": "RAM (random access memory) is fast read/write memory the CPU uses for running programs and data. It is volatile — cleared when power is lost. More RAM often helps multitasking and large datasets up to the point the workload fits.",
        "binary": "Computers represent data with bits (0 or 1). Eight bits make a byte. Binary arithmetic and Boolean logic (AND, OR, NOT, XOR) underlie circuits and processors. Hexadecimal is a compact human view of binary (four bits per hex digit).",
        "algorithm": "An algorithm is a finite sequence of well-defined steps to solve a problem. Good algorithms are correct, efficient in time and space, and clear to implement. Complexity is often expressed with Big-O notation for worst-case growth.",
        "programming": "Programming means writing instructions in a language a computer can run (directly or via translation). High-level languages (Python, JavaScript, Java, C++) trade human readability for translation steps like compiling or interpreting.",
        "python": "Python is a high-level language known for readable syntax and a large ecosystem. It is interpreted, dynamically typed, and widely used for scripting, web backends, data science, and education. Virtual environments isolate package versions per project.",
        "javascript": "JavaScript runs in browsers and on servers (Node.js). It is event-driven for web pages (DOM manipulation, async fetch). Modern JS includes classes, modules, and async/await for cleaner asynchronous code.",
        "html css": "HTML structures web content (headings, paragraphs, links, forms). CSS styles layout and appearance (selectors, box model, flexbox, grid). Together they separate content from presentation for maintainable web pages.",
        "internet": "The Internet is a global network of networks using standardized protocols. TCP/IP is the core suite: IP routes packets; TCP provides reliable ordered delivery; UDP is simpler and faster for many real-time uses. DNS maps names to IP addresses.",
        "http https": "HTTP is the application protocol for web requests and responses (stateless by default). HTTPS wraps HTTP in TLS encryption for confidentiality and integrity. Status codes (200 OK, 404 Not Found, 500 server error) communicate results.",
        "network router": "Routers forward packets between networks using IP addresses and routing tables. Home routers connect LAN devices to the ISP and often provide NAT, DHCP, Wi-Fi, and basic firewalling.",
        "ip address": "An IPv4 address is 32 bits, often written as four dotted decimals (e.g., 192.168.1.1). IPv6 uses 128-bit addresses for a vastly larger space. Private ranges are reused inside LANs; public addresses identify endpoints on the Internet with NAT in between.",
        "cybersecurity basics": "Core ideas: least privilege, strong passwords or passkeys, MFA, patching, encryption at rest and in transit, phishing awareness, and backups. Threat models balance usability with protection depending on asset value.",
        "database sql": "A database stores structured data with a schema. SQL is a declarative language for querying and modifying relational data (SELECT, JOIN, INSERT, UPDATE, DELETE). Indexes speed lookups at some write cost.",
        "git": "Git is a distributed version control system: commits are snapshots with hashes; branches isolate work; merges combine histories. Remote repositories (GitHub, GitLab) enable collaboration with pull requests and code review.",
        "api": "An API (application programming interface) defines how software components talk — often HTTP endpoints returning JSON for web APIs. REST uses resources and verbs; GraphQL lets clients request specific fields in one query.",
        "encryption": "Symmetric encryption uses one shared key (fast, key distribution hard). Asymmetric uses public/private key pairs for key exchange and signatures (slower). TLS combines them for secure channels on the web.",
        "operating system": "An OS schedules processes, manages memory, abstracts devices with drivers, and provides filesystems and security boundaries. Examples: Windows, macOS, Linux, Android, iOS. Kernels run in privileged mode; user programs run with restrictions.",
        "cloud computing": "Cloud delivers compute, storage, and services over the Internet with elastic scaling and pay-as-you-go models. IaaS offers VMs; PaaS adds managed runtimes; SaaS delivers full applications. Shared responsibility splits security duties between provider and customer.",
        "hostname dns naming": "A hostname is a label for a device or service on a network (e.g., www). DNS maps hostnames to IP addresses. A fully qualified domain name (FQDN) chains host labels plus the domain (www.example.com). Names follow DNS label rules; comparisons are usually case-insensitive. This is different from chemistry's systematic compound naming or everyday nicknames — say which domain you mean if you ask about a name.",
        "dns": "DNS (Domain Name System) is the Internet's directory: it maps human-readable names (like example.com) to IP addresses so routers can deliver traffic. Resolvers query in a hierarchy (root → TLD → authoritative servers). DNS also stores mail (MX) and other records; DNSSEC adds cryptographic authentication to reduce spoofing.",
        "tcp": "TCP (Transmission Control Protocol) is a connection-oriented transport protocol on top of IP. It provides reliable, ordered byte streams with acknowledgments, retransmissions, flow control, and congestion control. HTTP/HTTPS typically run over TCP; real-time apps sometimes prefer UDP when occasional loss is acceptable.",
        "udp": "UDP (User Datagram Protocol) is a lightweight, connectionless transport on IP. It sends datagrams without built-in reliability or ordering — good for DNS queries, VoIP, gaming, and streaming where low latency matters more than perfect delivery. Applications can add their own checks if needed.",
        "tls": "TLS (Transport Layer Security), often still called SSL, encrypts data between clients and servers — what powers HTTPS. It authenticates the server (and optionally the client) with certificates, negotiates cipher suites, and protects integrity so attackers cannot silently tamper. Always prefer HTTPS sites for passwords and payments.",
        "wifi": "Wi-Fi is wireless networking using IEEE 802.11 standards. Devices associate with an access point using radio channels in 2.4 GHz and/or 5 GHz (and 6 GHz for Wi-Fi 6E). Security evolved from WEP (broken) to WPA2/WPA3 with AES encryption. Throughput depends on channel width, MIMO antennas, interference, and distance.",
        "router": "A router connects networks and forwards packets toward their destination using IP addresses and routing tables. Home routers link your LAN to the ISP, usually perform NAT so many devices share one public IP, and often provide DHCP, DNS forwarding, Wi-Fi, and a basic firewall.",
        "firewall": "A firewall filters network traffic using rules (allow/deny) on addresses, ports, protocols, and sometimes application awareness. Host firewalls run on a device; network firewalls sit between zones. They implement defense in depth with least privilege — only open what you truly need.",
        "vpn": "A VPN (Virtual Private Network) creates an encrypted tunnel between your device and a VPN endpoint so traffic is hidden from local eavesdroppers on untrusted Wi-Fi. It changes where your traffic appears to exit on the Internet; it is not a substitute for HTTPS on sites or safe browsing by itself.",
        "variable and identifier naming": "In programming, names identify variables, functions, classes, modules, and keys. Languages enforce rules (reserved words, allowed characters, case sensitivity). Good names describe intent (user_count vs n). Namespaces and scopes prevent collisions. If you ask whether something has a name, specify language or snippet.",
        "default": "Good technology question! I cover middle school through college basics: how computers work, networks, the web, programming concepts, data, and security. Ask a specific term or scenario for a clear explanation.",
    },
}

KNOWLEDGE_BOOST_GLOB = "knowledge_boost*.json"


def merge_external_knowledge() -> None:
    """
    Merge optional external knowledge pack into KNOWLEDGE_BASE.
    File format:
    {
      "math": {"keyword": "answer", ...},
      "physics": {"keyword": "answer", ...}
    }
    """
    global _TOPIC_LEXICON
    boost_files = sorted(ROOT.glob(KNOWLEDGE_BOOST_GLOB))
    if not boost_files:
        return

    total_added = 0
    loaded_files = []
    for boost_path in boost_files:
        try:
            with open(boost_path, "r", encoding="utf-8") as f:
                extra = json.load(f)
        except Exception as e:
            print(f"[WARNING] Could not load {boost_path.name}: {e}")
            continue

        if not isinstance(extra, dict):
            print(f"[WARNING] {boost_path.name} must be a JSON object.")
            continue

        added = 0
        for subject, entries in extra.items():
            if subject == "greeting":
                continue
            if not isinstance(entries, dict):
                continue
            if subject not in KNOWLEDGE_BASE:
                label = subject.replace("_", " ").strip().title()
                KNOWLEDGE_BASE[subject] = {
                    "default": f"Ask a focused question about {label} "
                    "(middle school through college) and I will answer directly."
                }
            for keyword, answer in entries.items():
                if not isinstance(keyword, str) or not isinstance(answer, str):
                    continue
                key = keyword.strip().lower()
                val = answer.strip()
                if not key or not val:
                    continue
                if key not in KNOWLEDGE_BASE[subject]:
                    added += 1
                KNOWLEDGE_BASE[subject][key] = val

        if added:
            loaded_files.append(boost_path.name)
            total_added += added

    if total_added:
        print(
            f"[INFO] Loaded knowledge boost: +{total_added} entries from {len(loaded_files)} files"
        )
    _TOPIC_LEXICON = None


merge_external_knowledge()


def _kb_default_answer(subject: str) -> str:
    """Return the configured generic fallback string for a subject, if any."""
    kb = KNOWLEDGE_BASE.get(subject, {})
    d = kb.get("default")
    return d.strip() if isinstance(d, str) else ""


def _is_kb_default_answer(subject: str, answer: str) -> bool:
    """True when `answer` is the subject's generic KB scaffold (not a real fact snippet)."""
    if not answer or not isinstance(answer, str):
        return True
    d = _kb_default_answer(subject)
    if not d:
        return False
    return _normalize_text(answer) == _normalize_text(d)


SUBJECT_NAMES = {
    "american_history": "American History",
    "math":             "Mathematics",
    "english":          "English & Literature",
    "art":              "Art & Art History",
    "politics":         "Politics & Government",
    "biology":          "Biology",
    "physics":          "Physics",
    "chemistry":        "Chemistry",
    "technology":       "Technology & Computing",
    "greeting":         "Greeting",
}

GREETING_REPLIES = [
    "Hi — I'm your Academic Tutor. Ask me anything in American History, Math, English, Art, Politics, Biology, Physics, Chemistry, or Technology.",
    "Hello! I cover middle school through college in nine subject areas. What are you working on?",
    "Hey! I can help with history, math, English, art, government, biology, physics, chemistry, and computing — pick a topic and ask.",
]

THANKS_REPLIES = [
    "You're welcome — happy to help. Ask another question whenever you're ready.",
    "Any time! If something's still unclear, say which part and we can dig in.",
    "Glad that helped. Fire away with your next question when you want.",
]

BYE_REPLIES = [
    "Goodbye — good luck with your studies!",
    "See you later. You've got this.",
    "Bye for now. Come back anytime you want to keep learning.",
]

greeting_index = 0

FALLBACK_RESPONSES = [
    "That one's outside what I'm built for — I stick to American History, Math, English, Art, Politics, Biology, Physics, Chemistry, and Technology. Your teacher or a general reference might be the best next step.",
    "I'm not the right tutor for that topic. I do best with those nine academic areas (middle school through college level). Try rephrasing within them, or use a broader source for this one.",
]
fallback_index = 0

# Rotated with `_next_rotated` for natural variety (shares index with greetings).
CAPABILITY_SCOPE_REPLIES = [
    "I cover {subjects}. Ask me something specific in one of those — a concept, a problem, or a \"why does…\" — and I will answer directly.",
    "My wheelhouse is {subjects}. Pick one area and ask a concrete question (even a rough draft is fine), and we will work through it.",
    "I am here for {subjects} — middle school basics through advanced college. Paste your exact prompt when you can.",
]

CONVERSATION_REPAIR_REPLIES = [
    "That did not land — send the exact question again (one sentence is fine) and I will answer that directly.",
    "Let me refocus: paste the question you meant, and I will stick to it.",
    "Understood. What is the precise question you want answered next?",
]

SUBJECT_CLARIFY_OPTIONS = {
    "american_history": [
        "Founding era and Constitution",
        "Civil War and Reconstruction",
        "World Wars and Cold War",
        "Civil Rights and modern U.S. history",
    ],
    "math": [
        "Algebra and equations",
        "Geometry and trigonometry",
        "Calculus",
        "Statistics and probability",
    ],
    "english": [
        "Grammar and writing structure",
        "Literary devices and analysis",
        "Poetry and form",
        "Essay building and thesis support",
    ],
    "art": [
        "Art movements and history",
        "Drawing and composition",
        "Color theory and design",
        "Artists and key works",
    ],
    "politics": [
        "U.S. government structure",
        "Constitution and rights",
        "Elections and parties",
        "International relations",
    ],
    "biology": [
        "Cell biology and genetics",
        "Human body systems",
        "Evolution and ecology",
        "Biochemistry and molecular biology",
    ],
    "physics": [
        "Mechanics and motion",
        "Electricity and magnetism",
        "Waves, optics, and sound",
        "Modern physics (relativity/quantum)",
    ],
    "chemistry": [
        "Atoms, bonding, and the periodic table",
        "Reactions, stoichiometry, and gas laws",
        "Acids, bases, and solutions",
        "Introductory organic chemistry",
    ],
    "technology": [
        "How computers and programs work",
        "Internet, web, and networking basics",
        "Programming and algorithms",
        "Data, security, and cloud concepts",
    ],
}


def build_clarify_prompt(subject: str, intent_hints: Optional[List[str]] = None) -> str:
    label = SUBJECT_NAMES.get(subject, subject)
    options = SUBJECT_CLARIFY_OPTIONS.get(subject, [])
    hint_line = ""
    if intent_hints:
        intent_text = ", ".join(intent_hints[:2])
        hint_line = f" (I'm reading this as more of a {intent_text}-style question.)"
    if not options:
        return (
            f"I can help with {label}.{hint_line} Name a specific concept, problem, or reading — "
            "or paste your full assignment question — and I'll answer in one focused shot."
        )
    bullets = "\n".join(f"• {opt}" for opt in options)
    return (
        f"Sounds like {label}.{hint_line}\n\n"
        f"I can go deeper if you point me at one area:\n{bullets}\n\n"
        "Tap a chip below or paste your exact question — either works."
    )


def _supported_subject_labels() -> List[str]:
    ordered = [
        "american_history",
        "math",
        "english",
        "art",
        "politics",
        "biology",
        "physics",
        "chemistry",
        "technology",
    ]
    return [SUBJECT_NAMES[s] for s in ordered if s in SUBJECT_NAMES]


def _contains_keyword_phrase(q_norm: str, key_norm: str) -> bool:
    """
    Word-boundary phrase matching so short keys like "pi" do not match "topics".
    """
    if not q_norm or not key_norm:
        return False
    parts = [p for p in key_norm.split() if p]
    if not parts:
        return False
    pat = r"\b" + r"\s+".join(re.escape(p) for p in parts) + r"\b"
    return re.search(pat, q_norm) is not None


def match_meta_conversation(raw_text: str, core_text: str = "") -> Optional[Tuple[str, str, float, str]]:
    """
    Detect non-academic conversational intents (capabilities, correction feedback).
    """
    t = _normalize_text(raw_text)
    c = _normalize_text(core_text)
    merged = f"{t} {c}".strip()
    if not merged:
        return None

    capability_patterns = [
        r"\bwhat\s+(subjects|topics)\s+.*\byou\s+(help|cover|teach|support)\b",
        r"\bwhich\s+(subjects|topics)\s+.*\byou\s+(help|cover|teach|support)\b",
        r"\bwhat\s+can\s+you\s+(help|do|teach)\b",
        r"\bwhat\s+can\s+you\s+do\b",
        r"\b(your|the)\s+(scope|subjects|topics)\b",
        r"\bwhat\s+are\s+the\s+topics\s+you\s+help\s+with\b",
        r"\bwhat\s+do\s+you\s+help\s+with\b",
        r"\bwho\s+are\s+you\b",
        r"\bwhat\s+are\s+you\b",
        r"\bhow\s+does\s+this\s+work\b",
        r"\bwhat\s+is\s+this\b",
        r"^(help|help me|i need help)\b[\s!.?]*$",
    ]
    if any(re.search(p, merged) for p in capability_patterns):
        subject_text = ", ".join(_supported_subject_labels())
        reply = _next_rotated(CAPABILITY_SCOPE_REPLIES).format(subjects=subject_text)
        return reply, "greeting", 0.96, "capabilities_scope"

    correction_patterns = [
        r"\bi\s+did\s+not\s+ask\b",
        r"\bnot\s+what\s+i\s+asked\b",
        r"\bthat\s+is\s+wrong\b",
        r"\bthis\s+is\s+wrong\b",
        r"\byou\s+are\s+wrong\b",
        r"\bwrong\s+answer\b",
        r"\bincorrect\b",
        r"\bnot\s+my\s+question\b",
    ]
    if any(re.search(p, merged) for p in correction_patterns):
        reply = _next_rotated(CONVERSATION_REPAIR_REPLIES)
        return reply, "greeting", 0.9, "conversation_repair"

    return None


def _next_rotated(replies: list) -> str:
    global greeting_index
    r = replies[greeting_index % len(replies)]
    greeting_index += 1
    return r


def _looks_conversational_greeting(text: str) -> bool:
    """
    Catch conversational greetings such as:
    - "hello, how you doing"
    - "hey there how are you"
    - "hi how's it going today"
    and avoid routing them into academic semantic retrieval.
    """
    t = _normalize_text(text)
    if not t:
        return False

    words = t.split()
    if len(words) > 12:
        return False

    greeting_words = {"hi", "hello", "hey", "hiya", "howdy", "greetings", "yo"}
    if not any(w in greeting_words for w in words):
        return False

    conversational_words = {
        "how",
        "are",
        "you",
        "your",
        "doing",
        "do",
        "day",
        "today",
        "going",
        "there",
        "up",
        "hows",
        "whats",
        "sup",
        "good",
        "morning",
        "afternoon",
        "evening",
    }
    if not any(w in conversational_words for w in words):
        return False

    # If clearly academic terms are present, do not treat as small-talk.
    academic_hints = {
        "equation",
        "theorem",
        "history",
        "biology",
        "physics",
        "math",
        "literature",
        "politics",
        "art",
        "calculus",
        "war",
        "president",
        "constitution",
        "cell",
        "energy",
        "force",
        "power",
        "probability",
        "derivative",
        "chemistry",
        "technology",
        "programming",
        "network",
        "algorithm",
    }
    if any(w in academic_hints for w in words):
        return False

    return True


def match_greeting_or_smalltalk(text: str) -> Optional[Tuple[str, str, float]]:
    """
    Detect standalone greetings, thanks, or goodbyes. Returns (reply, 'greeting', 0.9) or None.
    """
    t = text.strip().lower()
    t = re.sub(r"\s+", " ", t)
    if not t or len(t) > 100:
        return None

    if _looks_conversational_greeting(t):
        subject_text = ", ".join(_supported_subject_labels())
        return (
            f"I am doing well — thanks for asking. I tutor {subject_text}. "
            "What would you like to dig into?",
            "greeting",
            0.93,
        )

    if re.match(
        r"^(hi|hello|hey|hiya|howdy|greetings|yo)\b"
        r"(\s+there|\s+again|\s+folks|\s+team|\s+all)?\s*[!?.,\s]*$",
        t,
        re.I,
    ):
        return _next_rotated(GREETING_REPLIES), "greeting", 0.9
    if re.match(
        r"^good\s+(morning|afternoon|evening|day)\b[\s!?.]*$",
        t,
        re.I,
    ):
        return _next_rotated(GREETING_REPLIES), "greeting", 0.9
    if re.match(
        r"^(how\s+are\s+you(\s+doing)?|how\s+you\s+doing|how('?s| is) it going|what'?s\s+up|whats\s+up|sup|wassup)\b[\s!?,.]*$",
        t,
        re.I,
    ):
        subject_text = ", ".join(_supported_subject_labels())
        return (
            f"Doing well, thanks. I am here for questions in {subject_text} — what is on your mind?",
            "greeting",
            0.9,
        )
    if re.match(
        r"^(thanks?|thank you|thx|ty|much appreciated|appreciate it|cheers)\b[\s!?.]*$",
        t,
        re.I,
    ):
        return _next_rotated(THANKS_REPLIES), "greeting", 0.9
    if re.match(
        r"^(bye|goodbye|see\s+you( later)?|later|gn|g' night|goodnight)\b[\s!?.]*$",
        t,
        re.I,
    ):
        return _next_rotated(BYE_REPLIES), "greeting", 0.9
    if re.match(r"^(ok|okay|k|cool|nice|great|got it|understood)\b[\s!?.]*$", t, re.I):
        return (
            "Sounds good — whenever you are ready, send your next question.",
            "greeting",
            0.85,
        )
    return None


def loose_pleasantry_fallback(text: str) -> Optional[Tuple[str, str, float]]:
    """
    Last-resort friendly reply for very short messages made only of common pleasantries
    (e.g. 'good morning', 'ok thanks') when the model is uncertain and no keyword matched.
    """
    t = text.strip().lower()
    if not t or len(t) > 45:
        return None
    words = re.findall(r"[a-z']+", t)
    if not words or len(words) > 5:
        return None
    if len(words) == 1 and words[0] in ("no", "nope", "nah", "why", "what"):
        return None
    allowed = frozenset(
        {
            "hi",
            "hello",
            "hey",
            "yo",
            "hiya",
            "howdy",
            "greetings",
            "bye",
            "goodbye",
            "thanks",
            "thank",
            "you",
            "thx",
            "ty",
            "cheers",
            "morning",
            "afternoon",
            "evening",
            "day",
            "good",
            "how",
            "are",
            "doing",
            "a",
            "the",
            "ok",
            "okay",
            "yes",
            "cool",
            "great",
            "nice",
            "pls",
            "please",
        }
    )
    if not all(w in allowed for w in words):
        return None
    if any(w in ("bye", "goodbye") for w in words):
        return _next_rotated(BYE_REPLIES), "greeting", 0.75
    if any(w in ("thanks", "thank", "thx", "ty", "cheers") for w in words):
        return _next_rotated(THANKS_REPLIES), "greeting", 0.75
    return _next_rotated(GREETING_REPLIES), "greeting", 0.75


def heuristic_answer(question: str, core: Optional[str] = None) -> Optional[Tuple[str, str, float, str]]:
    """
    Targeted intent heuristics for common paraphrases that keyword matching can miss.
    """
    # Arithmetic: keep raw text in the loop — `understand_question` strips '+' so core may lose '2+2'.
    raw_combo = f"{question} {core or ''}".lower()
    m_add = re.search(r"\b(\d{1,6})\s*\+\s*(\d{1,6})\b", raw_combo)
    if m_add:
        a, b = int(m_add.group(1)), int(m_add.group(2))
        total = a + b
        return (
            f"{a}+{b} equals {total}. (If you need a deeper explanation — for example carrying in base 10, "
            f"fractions, or modular arithmetic — say which level you are in.)",
            "math",
            0.94,
            "arithmetic_expression",
        )

    q = _normalize_text(question)
    tokens = set(q.split())

    # Ethics framed around slavery: history + civics (avoid misroutes to unrelated STEM snippets).
    if "slavery" in tokens and (
        "moral" in tokens or "ethics" in tokens or "ethical" in tokens or "wrong" in tokens
    ):
        cw = KNOWLEDGE_BASE["american_history"].get("civil war")
        pol = KNOWLEDGE_BASE["politics"].get("civil rights")
        if cw and pol:
            answer = (
                "Historians study slavery as a central institution in colonial and U.S. history — its laws, economics, "
                "resistance, and abolition — while civics and political philosophy ask how societies judge past wrongs and "
                "what obligations follow today. A balanced classroom answer usually separates **historical facts** "
                "(what happened, why, and who was affected) from **moral claims** (what should be condemned or repaired), "
                "then connects them clearly.\n\n"
                f"**American History lens:** {cw}\n\n**Politics / civics lens:** {pol}"
            )
            return answer, "american_history", 0.82, "slavery_ethics_framing"

    # "north vs south" style civil war paraphrases
    if {"north", "south"} <= tokens and ("fight" in tokens or "war" in tokens):
        answer = KNOWLEDGE_BASE["american_history"]["civil war"]
        return answer, "american_history", 0.65, "north_south_civil_war"

    # "first US leader" style first-president paraphrases
    if (
        ("leader" in tokens or "president" in tokens)
        and ("start" in tokens or "first" in tokens or "beginning" in tokens)
        and ("country" in tokens or "united" in tokens or "america" in tokens or "us" in tokens)
    ):
        answer = KNOWLEDGE_BASE["american_history"]["first president"]
        return answer, "american_history", 0.62, "first_us_leader"

    # speed vs velocity wording variations
    if "speed" in tokens and "velocity" in tokens:
        answer = KNOWLEDGE_BASE["physics"]["speed velocity"]
        return answer, "physics", 0.66, "speed_vs_velocity"

    return None


QUESTION_FILLER_PATTERNS = [
    r"\b(can you|could you|would you|please)\b",
    r"\b(i want to know|i need to know|help me understand)\b",
    r"\b(do you know|do u know)\b",
    r"\b(tell me|explain|describe|define)\b",
    r"\b(what can you tell me about)\b",
    r"\b(i was wondering)\b",
]


def understand_question(raw_text: str) -> str:
    """
    Build a clearer core query from informal phrasing.
    This lightweight "thinking layer" normalizes wording before intent routing.
    """
    q = _normalize_text(raw_text)
    if not q:
        return q

    # Remove common conversational wrappers.
    for pat in QUESTION_FILLER_PATTERNS:
        q = re.sub(pat, " ", q)
    q = re.sub(r"\b(about this|about that|for me|a bit|real quick)\b", " ", q)
    q = re.sub(r"\s+", " ", q).strip()

    # Normalize a few shorthand forms.
    q = q.replace("u s ", "us ").replace("u.s ", "us ")
    q = q.replace("pls ", "").replace("plz ", "")

    # Token typo correction using subject alias vocabulary (helps "physis" -> "physics").
    corrected_tokens: List[str] = []
    alias_tokens = {tok for alias in _subject_alias_vocab.keys() for tok in alias.split()}
    for tok in q.split():
        best_tok = tok
        best_score = 0.0
        for atok in alias_tokens:
            s = _similarity(tok, atok)
            if s > best_score:
                best_score = s
                best_tok = atok
        if best_score >= 0.88 and len(tok) >= 4:
            corrected_tokens.append(best_tok)
        else:
            corrected_tokens.append(tok)
    q = " ".join(corrected_tokens).strip()

    # Prefer trailing topical segment in long prompts:
    # e.g., "can you explain to me what the chain rule means in calculus"
    #  -> "what the chain rule means in calculus"
    m = re.search(r"\b(what|who|when|where|why|how)\b(.+)$", q)
    if m:
        q = (m.group(1) + m.group(2)).strip()

    # Convert "x in physics" into a clearer core target.
    q = re.sub(r"\bwhat is ([a-z0-9\s]+?) in ([a-z0-9\s]+)$", r"\1 \2", q).strip()
    q = re.sub(r"\s+", " ", q).strip()
    return q


def best_semantic_answer(
    original_question: str, core_question: str, preferred_subject: Optional[str] = None
) -> Optional[Tuple[str, str, float]]:
    """
    Try semantic retrieval on both original and core versions of the question,
    then return the strongest match.
    """
    candidates = []
    for q in (original_question, core_question):
        if not q:
            continue
        got = semantic_retrieval_answer(q, preferred_subject=preferred_subject)
        if got is not None:
            candidates.append(got)
    if not candidates:
        return None
    return max(candidates, key=lambda x: x[2])


def keyword_fallback_answer(question: str) -> Optional[Tuple[str, str]]:
    """
    If the model is uncertain, find any knowledge-base keyword contained in the question.
    Returns (subject_id, answer) or None.
    """
    q_norm = _normalize_text(question)
    q_tokens = set(q_norm.split())
    for subj, kb in KNOWLEDGE_BASE.items():
        for keyword, answer in kb.items():
            if keyword == "default":
                continue
            key_norm = _normalize_text(keyword)
            key_tokens = [t for t in key_norm.split() if t]
            direct = _contains_keyword_phrase(q_norm, key_norm)
            token_match = bool(key_tokens) and all(t in q_tokens for t in key_tokens)
            if direct or token_match:
                return subj, answer
    return None


def get_keyword_answer_for_subject(subject: str, question: str) -> Optional[str]:
    """Return a direct keyword match answer for a subject, or None."""
    q_norm = _normalize_text(question)
    q_tokens = set(q_norm.split())
    kb = KNOWLEDGE_BASE.get(subject, {})
    for keyword, answer in kb.items():
        if keyword == "default":
            continue
        key_norm = _normalize_text(keyword)
        key_tokens = [t for t in key_norm.split() if t]
        direct = _contains_keyword_phrase(q_norm, key_norm)
        token_match = bool(key_tokens) and all(t in q_tokens for t in key_tokens)
        if direct or token_match:
            return answer
    return None


def get_answer(subject: str, question: str) -> str:
    """Look up the best matching answer in the knowledge base for a given subject."""
    kb = KNOWLEDGE_BASE.get(subject, {})
    keyword_answer = get_keyword_answer_for_subject(subject, question)
    if keyword_answer is not None:
        return keyword_answer
    return kb.get(
        "default",
        f"Good {SUBJECT_NAMES.get(subject, subject)} question — I will need a bit more detail. "
        "Name a person, event, concept, or problem (or paste the full prompt) and I will answer in one focused pass.",
    )


def _confidence_band(model_conf: float, margin: Optional[float] = None) -> str:
    """Human-readable band for model softmax (do not treat as semantic cosine)."""
    m = margin if margin is not None else 1.0
    if model_conf >= 0.78 and m >= 0.10:
        return "high"
    if model_conf >= 0.55 or (model_conf >= 0.45 and m >= 0.06):
        return "medium"
    return "low"


def _with_subject_frame(subject: str, answer: str) -> str:
    """Layer 3: keep answers anchored to the routed subject (light framing)."""
    if subject in ("greeting", "out_of_scope"):
        return answer
    label = SUBJECT_NAMES.get(subject, subject)
    return f"**{label}** — {answer}"


# ── Layer 1: strict off-topic gate (only obvious non-academic; when in doubt, stay in-scope) ──
_OFF_TOPIC_PATTERNS = [
    re.compile(r"\b(best|easy)\s+(recipe|recipes)\b"),
    re.compile(r"\b(how\s+to\s+cook|baking\s+a|ingredients\s+for)\b"),
    re.compile(r"\b(super\s+bowl|nba\s+finals|nfl\s+draft|world\s+cup\s+odds)\b"),
    re.compile(r"\b(fantasy\s+football|betting\s+odds|sportsbook)\b"),
    re.compile(r"\b(netflix|disney\s*\+|hulu)\s+(show|series)\b"),
    re.compile(r"\b(where\s+to\s+buy|cheapest\s+iphone|amazon\s+deal)\b"),
    re.compile(r"\b(write\s+my\s+essay\s+for\s+me|do\s+my\s+homework\s+for\s+money)\b"),
    re.compile(r"\b(hack\s+into|how\s+to\s+steal|buy\s+illegal)\b"),
]


def clearly_off_topic(raw: str, core: str) -> Tuple[bool, str]:
    """True only for clear non-tutor topics (per NLU pipeline scope gate)."""
    m = _normalize_text(f"{raw} {core}")
    if not m:
        return False, ""
    for pat in _OFF_TOPIC_PATTERNS:
        if pat.search(m):
            return True, pat.pattern
    return False, ""


# ── Layer 2: lightweight pedagogical subject hints (regex; complements the DL classifier) ──
_NLU_SUBJECT_RULES: List[Tuple[re.Pattern[str], str, float, str]] = [
    (
        re.compile(
            r"\b(noun|pronoun|verb|adjective|adverb|preposition|conjunction|interjection|grammar|syntax|"
            r"comma\s+splice|sentence\s+fragment|metaphor|simile|alliteration|hyperbole|oxymoron|"
            r"thesis\s+statement|topic\s+sentence|rhetorical|figurative\s+language|literary\s+device|"
            r"point\s+of\s+view|stanza|sonnet|haiku|iambic|rhyme\s+scheme|essay|mla|citation|plagiarism)\b"
        ),
        "english",
        0.82,
        "grammar_or_writing_cues",
    ),
    (
        re.compile(
            r"\b(algebra|geometry|calculus|trigonometry|derivative|integral|polynomial|quadratic|"
            r"logarithm|fraction|decimal|equation|triangle|circle|slope|perpendicular|matrix|vector|"
            r"probability|statistics|pythagorean|hypotenuse|exponent|factorial|arithmetic|add|subtract|"
            r"multiply|divide|equals|plus|minus|sum|digit|integer|rational\s+number)\b|\d+\s*\+\s*\d+"
        ),
        "math",
        0.82,
        "math_cues",
    ),
    (
        re.compile(
            r"\b(photosynthesis|mitosis|meiosis|dna|rna|cell|enzyme|protein|gene|allele|ecosystem|"
            r"evolution|natural\s+selection|osmosis|anatomy|physiology|bacteria|virus|vaccine)\b"
        ),
        "biology",
        0.8,
        "biology_cues",
    ),
    (
        re.compile(
            r"\b(newton|einstein|velocity|acceleration|momentum|force|energy|joule|watt|electric|magnetic|"
            r"photon|quantum|relativity|gravity|thermodynamics|entropy|wave|frequency|ohm|circuit|"
            r"mass.?energy|e\s*=\s*mc|mc\^?2|special\s+relativity|light\s+speed)\b"
        ),
        "physics",
        0.8,
        "physics_cues",
    ),
    (
        re.compile(
            r"\b(president|declaration\s+of\s+independence|constitution|civil\s+war|revolution|"
            r"gettysburg|emancipation|slavery|abolition|founding\s+fathers|amendment|impeachment|"
            r"cold\s+war|civil\s+rights|pearl\s+harbor|bill\s+of\s+rights)\b"
        ),
        "american_history",
        0.81,
        "us_history_cues",
    ),
    (
        re.compile(
            r"\b(democracy|republic|senate|congress|election|vote|ballot|gerrymander|filibuster|"
            r"supreme\s+court|federalism|bill\s+of\s+rights|executive\s+branch|legislative|judicial\s+review)\b"
        ),
        "politics",
        0.78,
        "civics_cues",
    ),
    (
        re.compile(
            r"\b(painting|sculpture|renaissance|impressionism|cubism|baroque|palette|canvas|"
            r"perspective|museum|drawing|color\s+theory|composition)\b"
        ),
        "art",
        0.76,
        "art_cues",
    ),
    (
        re.compile(
            r"\b(periodic\s+table|element|compound|molecule|atom|ion|ionic|covalent|acid|base|ph\b|"
            r"mole\b|stoichiometry|oxidation|reduction|redox|catalyst|enthalpy|isotope|electron\s+shell|"
            r"valence|solution|solubility|organic\s+chemistry|hydrocarbon|bonding)\b"
        ),
        "chemistry",
        0.8,
        "chemistry_cues",
    ),
    (
        re.compile(
            r"\b(python|javascript|java\b|programming|algorithm|computer|cpu|ram|binary|database|sql|"
            r"html|css|internet|network|router|routers|switch|ethernet|wifi|wireless|wan|lan|"
            r"http|https|tcp|udp|tls|ssl|vpn|firewall|ip\b|ipv6|ipv4|dns|dhcp|nat|server|git|github|"
            r"api|encryption|cybersecurity|operating\s+system|linux|windows|software|hardware|"
            r"cloud|virtual\s+machine|compiler|debug|kubernetes|docker|oauth)\b"
        ),
        "technology",
        0.78,
        "technology_cues",
    ),
]


def nlu_pedagogical_subject(raw: str, core: str) -> Optional[Tuple[str, float, str]]:
    """
    Layer 2 helper: suggest subject from academic wording when the classifier is weak.
    Returns (subject_key, pseudo_confidence, reason_tag) or None.
    """
    m = _normalize_text(f"{raw} {core}")
    if not m:
        return None
    plain = f"{raw} {core}".lower()
    # Keep + and = for arithmetic / famous equations (_normalize_text strips them).
    if re.search(r"\d+\s*\+\s*\d+", plain):
        return "math", 0.84, "arithmetic_expression"
    if re.search(r"e\s*=\s*mc\^?2|e\s*=\s*mc2|mc\^?2", plain, re.I):
        return "physics", 0.84, "mass_energy_equation"
    best: Optional[Tuple[str, float, str]] = None
    for pat, subj, base_score, tag in _NLU_SUBJECT_RULES:
        if pat.search(m):
            if best is None or base_score > best[1]:
                best = (subj, base_score, tag)
    return best


_CROSS_DOMAIN_TRIGGERS = [
    re.compile(r"\brelationship between\b"),
    re.compile(r"\bconnection between\b"),
    re.compile(r"\bhow does .{4,90} relate to\b"),
    re.compile(r"\bcompare .{4,80} and\b"),
    re.compile(r"\bcompare .{4,80} to\b"),
    re.compile(r"\binterdisciplinary\b"),
    re.compile(r"\bphysics and chemistry\b"),
    re.compile(r"\bphysics\s+vs\s+chemistry\b"),
    re.compile(r"\bchemistry\s+vs\s+physics\b"),
    re.compile(r"\bchemistry and biology\b"),
    re.compile(r"\bmath and physics\b"),
    re.compile(r"\bhistory and politics\b"),
]


def is_cross_domain_question(raw: str, core: str) -> bool:
    """Heuristic: user explicitly asks to connect two domains (NLU pipeline multi-subject path)."""
    t = _normalize_text(f"{raw} {core}")
    if not t or len(t) < 18:
        return False
    return any(p.search(t) for p in _CROSS_DOMAIN_TRIGGERS)


def synthesize_cross_subject_answer(
    raw: str, core: str, ranked_probs: List[Tuple[str, float]]
) -> Optional[str]:
    """Build a short combined answer from the top two classifier subjects when cues match."""
    if len(ranked_probs) < 2:
        return None
    s1, p1 = ranked_probs[0]
    s2, p2 = ranked_probs[1]
    if s1 == s2 or p2 < 0.11:
        return None
    q = f"{raw} {core}".strip()
    chunks: List[str] = []
    for subj in (s1, s2):
        if subj not in KNOWLEDGE_BASE:
            continue
        hit = get_keyword_answer_for_subject(subj, q)
        if hit:
            label = SUBJECT_NAMES.get(subj, subj)
            chunks.append(f"**{label}:** {hit}")
        else:
            fb = KNOWLEDGE_BASE.get(subj, {}).get("default")
            if fb:
                label = SUBJECT_NAMES.get(subj, subj)
                chunks.append(f"**{label} (overview):** {fb[:380]}…")
    if len(chunks) < 2:
        return None
    return (
        "This question bridges more than one subject. Here is how each area contributes — "
        "always check your teacher's framing for interdisciplinary assignments:\n\n"
        + "\n\n".join(chunks)
    )


def predict_subject_details(text: str) -> Tuple[str, float, Dict[str, float]]:
    """Return (subject, confidence, top_probabilities) for an input string."""
    if not is_ready():
        raise RuntimeError("Model not loaded. Call load_artifacts() or run train.py first.")
    with _predict_lock:
        seq = tokenizer.texts_to_sequences([text.lower()])
        padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
        probs = model.predict(padded, verbose=0)[0]
        idx = int(np.argmax(probs))
        ranked = sorted(
            [(le.classes_[i], float(probs[i])) for i in range(len(probs))],
            key=lambda x: x[1],
            reverse=True,
        )
        top = {k: round(v, 4) for k, v in ranked[:3]}
        return le.classes_[idx], float(probs[idx]), top


def predict_subject(text: str):
    """Return (subject, confidence) for a given input string."""
    subject, confidence, _ = predict_subject_details(text)
    return subject, confidence


def semantic_debug(question: str, preferred_subject: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Return semantic retrieval result + score for debug visibility."""
    sem = semantic_retrieval_answer(question, preferred_subject=preferred_subject)
    if sem is None:
        return None
    subj, answer, score = sem
    return {
        "subject": subj,
        "score": round(float(score), 4),
        "answer_preview": answer[:140],
    }


def chat_with_debug(
    user_input: str,
    chat_context: Optional[Dict[str, Any]] = None,
) -> Tuple[str, str, float, Dict[str, Any]]:
    global fallback_index
    ctx = chat_context or {}
    pending_followup = bool(ctx.get("last_route") == "reasoning_clarify")
    prior_thread = (ctx.get("last_clarify_thread") or "").strip()
    reasoning_raw = (
        f"{prior_thread} {user_input}".strip()
        if pending_followup and prior_thread
        else user_input.strip()
    )

    core_input = understand_question(user_input)
    subject_hint = detect_subject_hint(f"{user_input} {core_input}".strip())
    intent_hints = detect_intent_hints(f"{user_input} {core_input}".strip())
    low_info = is_low_information_query(core_input or user_input)
    negated_subjects = detect_negated_subjects(f"{user_input} {core_input}")
    if subject_hint is not None and subject_hint[0] in negated_subjects:
        subject_hint = None
    explicit_subject_ref = (
        subject_hint is not None
        and has_explicit_subject_reference(f"{user_input} {core_input}", subject_hint[0])
    )
    debug: Dict[str, Any] = {
        "input_original": user_input,
        "input_core": core_input,
        "reasoning_context": {
            "pending_followup": pending_followup,
            "reasoning_raw_preview": (reasoning_raw[:200] + "...") if len(reasoning_raw) > 200 else reasoning_raw,
        },
        "route": "unknown",
        "subject_hint": (
            {
                "subject": subject_hint[0],
                "score": round(subject_hint[1], 4),
                "matched_terms": subject_hint[2][:6],
            }
            if subject_hint
            else None
        ),
        "intent_hints": intent_hints,
        "low_information_query": low_info,
        "negated_subjects": negated_subjects,
        "explicit_subject_reference": explicit_subject_ref,
        "confidence_band": None,
    }

    # Standalone "hi", "thanks", etc. before the classifier
    st = match_greeting_or_smalltalk(user_input)
    if st is not None:
        debug["route"] = "greeting_smalltalk"
        debug["greeting_detected"] = True
        return st[0], st[1], st[2], debug
    debug["greeting_detected"] = False

    meta = match_meta_conversation(user_input, core_input)
    if meta is not None:
        debug["route"] = meta[3]
        return meta[0], meta[1], meta[2], debug

    off, off_pat = clearly_off_topic(user_input, core_input or "")
    if off:
        debug["route"] = "scope_gate_off_topic"
        debug["scope_gate"] = {"in_scope": False, "reason": "non_academic_pattern", "pattern": off_pat}
        response = FALLBACK_RESPONSES[fallback_index % len(FALLBACK_RESPONSES)]
        fallback_index += 1
        return response, "out_of_scope", 0.22, debug

    heur = heuristic_answer(user_input, core_input or "")
    if heur is not None:
        debug["route"] = "heuristic"
        debug["heuristic_rule"] = heur[3]
        return heur[0], heur[1], heur[2], debug

    # Subject-aware clarification for very short/ambiguous prompts.
    if (
        subject_hint is not None
        and subject_hint[1] >= 0.70
        and subject_hint[0] not in negated_subjects
        and (low_info or explicit_subject_ref)
    ):
        hinted_subject = subject_hint[0]
        core_tokens = _tokenize_words(core_input or user_input)
        hinted_default = KNOWLEDGE_BASE.get(hinted_subject, {}).get("default")
        # Clarify mode only for underspecified prompts.
        # If user already gave an intent (e.g., "explain geometry") do not re-clarify.
        # If the user already signaled a define/explain/how intent, do not treat token-short cores
        # (e.g. after "what is X in chemistry" rewriting) as underspecified subject-only prompts.
        should_clarify = (low_info and not intent_hints) or (
            explicit_subject_ref and len(core_tokens) <= 5 and not intent_hints
        )
        near_default = bool(
            hinted_default
            and explicit_subject_ref
            and len(core_tokens) <= 4
            and len(core_tokens) >= 2
            and not intent_hints
        )
        # Replace only the generic "Good <subject> question…" stub when retrieval is solid.
        if near_default:
            sem_hint_early = best_semantic_answer(
                user_input, core_input or "", preferred_subject=hinted_subject
            )
            if sem_hint_early is not None:
                hs_subj, hs_ans, hs_score = sem_hint_early
                if hs_score >= SEMANTIC_WEAK_THRESHOLD and not _is_kb_default_answer(hs_subj, hs_ans):
                    debug["route"] = "subject_hint_semantic_early"
                    debug["semantic_threshold"] = SEMANTIC_WEAK_THRESHOLD
                    return hs_ans, hs_subj, max(float(subject_hint[1]), hs_score), debug
        if should_clarify:
            debug["route"] = "subject_clarify_mode"
            clarify = build_clarify_prompt(hinted_subject, intent_hints=intent_hints)
            return clarify, hinted_subject, max(0.68, subject_hint[1]), debug
        if near_default and hinted_default:
            debug["route"] = "subject_hint_default"
            return hinted_default, hinted_subject, max(0.62, subject_hint[1]), debug

    # Strong semantic match: answer immediately, even before classifier confidence.
    sem = best_semantic_answer(user_input, core_input)
    debug["semantic_original"] = semantic_debug(user_input)
    debug["semantic_core"] = semantic_debug(core_input) if core_input else None
    if sem is not None:
        sem_subject, sem_answer, sem_score = sem
        if sem_score >= SEMANTIC_STRONG_THRESHOLD:
            debug["route"] = "semantic_strong"
            debug["semantic_threshold"] = SEMANTIC_STRONG_THRESHOLD
            return sem_answer, sem_subject, sem_score, debug

    subject, confidence, top_probs = predict_subject_details(core_input or user_input)
    ranked_probs = sorted(top_probs.items(), key=lambda x: x[1], reverse=True)
    subject_margin = (
        float(ranked_probs[0][1] - ranked_probs[1][1]) if len(ranked_probs) > 1 else 1.0
    )
    debug["subject_margin"] = round(subject_margin, 4)
    if (
        subject_hint is not None
        and subject_hint[1] >= 0.82
        and subject_hint[0] not in negated_subjects
        and (confidence < 0.92 or explicit_subject_ref)
    ):
        hinted_subject = subject_hint[0]
        if hinted_subject != subject:
            debug["subject_override"] = {
                "from": subject,
                "to": hinted_subject,
                "reason": "strong_subject_hint_explicit" if explicit_subject_ref else "strong_subject_hint",
            }
            subject = hinted_subject
            confidence = max(confidence, min(0.9, subject_hint[1]))
    # Mass-energy identity: keep routing on physics even if the embedding classifier misfires.
    plain_eq = f"{user_input} {core_input or ''}".lower()
    if re.search(r"e\s*=\s*mc\^?2|e\s*=\s*mc2|mc\^?2", plain_eq, re.I):
        if subject != "physics":
            debug["classifier_physics_equation_override"] = {"from": subject, "to": "physics"}
        subject = "physics"
        confidence = max(float(confidence), 0.88)
    # Classifier sometimes labels obvious computing questions as physics/chemistry (TLS, Wi-Fi, TCP, …).
    nlu_tech_chk = nlu_pedagogical_subject(user_input, core_input or "")
    if (
        nlu_tech_chk is not None
        and nlu_tech_chk[0] == "technology"
        and nlu_tech_chk[2] == "technology_cues"
        and subject != "technology"
    ):
        stem_misfire = subject in ("physics", "chemistry", "biology", "math")
        if stem_misfire or float(confidence) < 0.84 or float(subject_margin) < 0.11:
            debug["technology_cue_override"] = {
                "from": subject,
                "to": "technology",
                "reason": nlu_tech_chk[2],
                "classifier_confidence": round(float(confidence), 4),
            }
            subject = "technology"
            confidence = max(float(confidence), min(0.86, float(nlu_tech_chk[1])))
    debug["predicted_subject"] = subject
    debug["predicted_confidence"] = round(confidence, 4)
    debug["top_subject_probabilities"] = top_probs
    debug["confidence_band"] = _confidence_band(float(confidence), subject_margin)

    # Multi-subject synthesis when the question explicitly asks to connect domains (NLU pipeline).
    if (
        is_cross_domain_question(user_input, core_input or "")
        and len(ranked_probs) >= 2
        and ranked_probs[0][0] != ranked_probs[1][0]
        and subject_margin < 0.20
    ):
        ranked_tuples = [(str(k), float(v)) for k, v in ranked_probs]
        synth_text = synthesize_cross_subject_answer(
            user_input, core_input or "", ranked_tuples
        )
        if synth_text:
            debug["route"] = "multi_subject_synthesis"
            debug["synthesis_subjects"] = [ranked_tuples[0][0], ranked_tuples[1][0]]
            syn_conf = max(ranked_tuples[0][1], ranked_tuples[1][1], 0.55)
            return synth_text, ranked_tuples[0][0], float(syn_conf), debug

    # High-confidence classifier path with better fallback than subject default.
    if confidence >= CONFIDENCE_THRESHOLD:
        direct_answer = get_keyword_answer_for_subject(subject, core_input or user_input)
        kw_subject = subject
        if (
            direct_answer is None
            and len(ranked_probs) > 1
            and subject_margin < 0.11
            and ranked_probs[1][1] >= 0.12
        ):
            alt_subj = ranked_probs[1][0]
            if alt_subj != subject:
                da2 = get_keyword_answer_for_subject(alt_subj, core_input or user_input)
                if da2 is not None:
                    direct_answer = da2
                    kw_subject = alt_subj
                    debug["subject_second_place_try"] = alt_subj
        # Layer 2: when the model is split, prefer strong pedagogical cues over the top label.
        if direct_answer is None:
            nlu_hi = nlu_pedagogical_subject(user_input, core_input or "")
            if (
                nlu_hi is not None
                and nlu_hi[0] != subject
                and subject_margin < 0.085
                and nlu_hi[1] >= 0.79
            ):
                da_n = get_keyword_answer_for_subject(nlu_hi[0], core_input or user_input)
                if da_n is not None:
                    direct_answer = da_n
                    kw_subject = nlu_hi[0]
                    debug["nlu_router"] = {"subject": kw_subject, "reason": nlu_hi[2], "note": "high_conf_margin_override"}
        if direct_answer is not None:
            debug["route"] = "classifier_keyword"
            return direct_answer, kw_subject, confidence, debug

        merged_q = f"{user_input} {core_input or ''}".strip()
        if keyword_fallback_answer(merged_q) is None and keyword_fallback_answer(
            core_input or user_input
        ) is None:
            r_hi = reasoning_layers_assess(
                reasoning_raw,
                core_input or "",
                subject,
                subject_hint,
                negated_subjects,
                float(confidence),
                float(subject_margin),
                pending_followup_clarify=pending_followup,
                current_turn_text=user_input,
            )
            debug["reasoning"] = r_hi
            if r_hi.get("clarify"):
                debug["route"] = "reasoning_clarify"
                return (
                    build_reasoning_clarify_response(subject),
                    subject,
                    min(float(confidence), 0.58),
                    debug,
                )

        sem_pref = best_semantic_answer(user_input, core_input, preferred_subject=subject)
        if sem_pref is not None:
            sem_subject, sem_answer, sem_score = sem_pref
            if sem_score >= SEMANTIC_WEAK_THRESHOLD:
                if sem_score < SEMANTIC_TRUST_SCORE:
                    r_sem = reasoning_layers_assess(
                        reasoning_raw,
                        core_input or "",
                        subject,
                        subject_hint,
                        negated_subjects,
                        float(confidence),
                        float(subject_margin),
                        pending_followup_clarify=pending_followup,
                        current_turn_text=user_input,
                    )
                    debug["reasoning_post_semantic"] = r_sem
                    if r_sem.get("clarify"):
                        debug["route"] = "reasoning_clarify"
                        return (
                            build_reasoning_clarify_response(subject),
                            subject,
                            min(max(float(sem_score), float(confidence)), 0.58),
                            debug,
                        )
                debug["route"] = "classifier_semantic_preferred"
                debug["semantic_threshold"] = SEMANTIC_WEAK_THRESHOLD
                return sem_answer, sem_subject, max(confidence, sem_score), debug

        answer = get_answer(subject, core_input or user_input)
        debug["route"] = "classifier_default_subject_answer"
        return _with_subject_frame(subject, answer), subject, confidence, debug

    if confidence < CONFIDENCE_THRESHOLD:
        kw = keyword_fallback_answer(core_input or user_input)
        if kw is not None:
            subj, answer = kw
            debug["route"] = "low_conf_keyword_fallback"
            return answer, subj, max(confidence, 0.55), debug

        sem_any = best_semantic_answer(user_input, core_input)
        if sem_any is not None:
            sem_subject, sem_answer, sem_score = sem_any
            if sem_score >= SEMANTIC_WEAK_THRESHOLD:
                if sem_score < SEMANTIC_TRUST_SCORE:
                    r_lsem = reasoning_layers_assess(
                        reasoning_raw,
                        core_input or "",
                        subject,
                        subject_hint,
                        negated_subjects,
                        float(confidence),
                        float(subject_margin),
                        pending_followup_clarify=pending_followup,
                        current_turn_text=user_input,
                    )
                    debug["reasoning_post_semantic_low"] = r_lsem
                    if r_lsem.get("clarify"):
                        debug["route"] = "reasoning_clarify"
                        return (
                            build_reasoning_clarify_response(subject),
                            subject,
                            max(float(sem_score), 0.48),
                            debug,
                        )
                debug["route"] = "low_conf_semantic_fallback"
                debug["semantic_threshold"] = SEMANTIC_WEAK_THRESHOLD
                return sem_answer, sem_subject, sem_score, debug

        lp = loose_pleasantry_fallback(core_input or user_input)
        if lp is not None:
            debug["route"] = "low_conf_pleasantry_fallback"
            return lp[0], lp[1], lp[2], debug

        # Layer 2: pedagogical NLU before declaring out-of-scope (stay in-scope when uncertain).
        nlu_low = nlu_pedagogical_subject(user_input, core_input or "")
        if nlu_low is not None:
            subj_n, sc_n, tag_n = nlu_low
            debug["nlu_router"] = {"subject": subj_n, "score": sc_n, "reason": tag_n}
            ans_n = get_answer(subj_n, core_input or user_input)
            debug["route"] = "nlu_pedagogical_router"
            debug["scope_gate"] = {"in_scope": True, "reason": "pedagogical_cue"}
            return _with_subject_frame(subj_n, ans_n), subj_n, max(float(confidence), sc_n), debug

        merged_lo = f"{user_input} {core_input or ''}".strip()
        if keyword_fallback_answer(merged_lo) is None and keyword_fallback_answer(
            core_input or user_input
        ) is None:
            r_lo = reasoning_layers_assess(
                reasoning_raw,
                core_input or "",
                subject,
                subject_hint,
                negated_subjects,
                float(confidence),
                float(subject_margin),
                pending_followup_clarify=pending_followup,
                current_turn_text=user_input,
            )
            debug["reasoning"] = r_lo
            if r_lo.get("clarify"):
                debug["route"] = "reasoning_clarify"
                return (
                    build_reasoning_clarify_response(subject),
                    subject,
                    max(float(confidence), 0.48),
                    debug,
                )

        # Layer 1: when still uncertain, prefer the model's best subject over out-of-scope (skill: in doubt → in-scope).
        ans_in = get_answer(subject, core_input or user_input)
        debug["route"] = "low_conf_in_scope_default"
        debug["scope_gate"] = {"in_scope": True, "reason": "uncertain_but_academic_default"}
        return _with_subject_frame(subject, ans_in), subject, max(0.52, float(confidence)), debug

    # Safety fallback (should not be reached due branching above).
    answer = get_answer(subject, user_input)
    debug["route"] = "safety_fallback"
    return answer, subject, confidence, debug


def chat(user_input: str) -> Tuple[str, str, float]:
    response, subject, confidence, _debug = chat_with_debug(user_input, None)
    return response, subject, confidence


# ── CLI Chat Loop ─────────────────────────────────────────────────────────────
def main():
    err = load_artifacts()
    if err:
        print(f"[ERROR] {err}")
        return
    print("[INFO] Academic Tutor ready!\n")
    subjects_list = ", ".join(SUBJECT_NAMES.values())
    print("=" * 65)
    print("   Academic Tutor Chatbot  (Deep Learning)")
    print(f"   Subjects: {subjects_list}")
    print("   Type 'quit' or 'exit' to stop")
    print("=" * 65)
    print()

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit", "bye"}:
            print("Tutor: Goodbye! Keep studying — you've got this!")
            break

        response, subject, confidence = chat(user_input)
        label = SUBJECT_NAMES.get(subject, subject)
        print(f"\nTutor: {response}")
        print(f"       [subject: {label} | confidence: {confidence:.2%}]\n")


if __name__ == "__main__":
    main()
