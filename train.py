"""
Deep Learning Academic Tutor Chatbot - Training Script
Subjects: American History, Math, English, Art, Politics, Biology, Physics,
          Chemistry, Technology & Computing
Level: Middle school through advanced college
Architecture: Embedding + Conv1D n-gram features + dense head (see academic-tutor-dl-training skill).

MAX_LEN must stay aligned with chatbot.py (pad_sequences at inference).
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv1D,
    Dense,
    Dropout,
    Embedding,
    GlobalMaxPooling1D,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from chatbot import KNOWLEDGE_BASE

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ── Hyperparameters (Conv1D tier — skill reference.md "Architecture options A") ─
# MAX_LEN kept in sync with chatbot.py
VOCAB_SIZE     = 5000
MAX_LEN        = 20
EMBEDDING_DIM  = 128
EPOCHS         = 100
BATCH_SIZE     = 32
TEST_SIZE      = 0.15
VAL_SIZE       = 0.15
LEARNING_RATE  = 0.001
EARLY_PATIENCE = 15
LR_PATIENCE    = 6

# ── Paths (work from any working directory) ─────────────────────────────────
ROOT = Path(__file__).resolve().parent
CHECKPOINT_DIR = ROOT / "checkpoints"
CLASSIFIER_SUBJECT_KEYS = frozenset(k for k in KNOWLEDGE_BASE if k != "greeting")

AUGMENTATION_TEMPLATES: List[str] = [
    "{keyword}",
    "what is {keyword}",
    "explain {keyword}",
    "tell me about {keyword}",
    "can you explain {keyword}",
    "help me understand {keyword}",
    "i do not get {keyword}",
    "quiz tomorrow on {keyword}",
    "teacher asked about {keyword}",
    "need a simple explanation of {keyword}",
    "what does {keyword} mean",
    "how does {keyword} work",
    "why is {keyword} important",
    "define {keyword}",
    "describe {keyword}",
    "study guide for {keyword}",
    "exam question about {keyword}",
    "i am confused about {keyword}",
    "struggling with {keyword}",
    "difference between {keyword} and",
    "compare {keyword} to",
    "regarding {keyword}",
]


def print_data_audit(df: pd.DataFrame, max_len: int) -> None:
    """Phase 1 — data health before training (academic-tutor-dl-training)."""
    vc = df["intent"].value_counts()
    ratio = float(vc.max() / vc.min()) if vc.min() > 0 else float("inf")
    print("\n" + "=" * 60)
    print("  Data audit (pre-train)")
    print("=" * 60)
    print("=== Class distribution ===")
    print(vc.to_string())
    print(f"\nSmallest class: {int(vc.min())} samples")
    print(f"Largest class:  {int(vc.max())} samples")
    print(f"Imbalance ratio: {ratio:.2f}x (skill: consider rebalancing if > ~3x)")
    dup_q = df[df.duplicated(subset=["question"], keep=False)]
    print(f"\n=== Duplicate questions (any intent): {len(dup_q)} rows ===")
    if len(dup_q) > 0:
        print(dup_q[["question", "intent"]].head(8).to_string(index=False))
    lengths = df["question"].astype(str).str.split().str.len()
    print(f"\n=== Question length (words) ===")
    print(lengths.describe().to_string())
    over = int((lengths > max_len).sum())
    print(f"Questions longer than MAX_LEN={max_len} (truncated when padded): {over}")
    short = df[lengths <= 2]
    print(f"\n=== Very short questions (<=2 words): {len(short)} ===")
    if len(short) > 0:
        print(short[["question", "intent"]].head(12).to_string(index=False))
    unknown = sorted(set(df["intent"].unique()) - CLASSIFIER_SUBJECT_KEYS)
    if unknown:
        print(f"\n[WARNING] Intents not in KNOWLEDGE_BASE (non-greeting): {unknown}")
    print("=" * 60 + "\n")


def validate_dataset_labels(df: pd.DataFrame) -> list[str]:
    """Reject unknown intent labels vs knowledge-base subject keys."""
    errors: list[str] = []
    for intent in df["intent"].unique():
        s = str(intent).strip()
        if s not in CLASSIFIER_SUBJECT_KEYS:
            errors.append(f"Unknown intent label: {s!r}")
    return errors


def build_augmented_examples() -> pd.DataFrame:
    """
    Create extra training questions from the knowledge base so the classifier
    sees many more ways users might ask for the same topic.
    """
    subject_teasers = {
        "american_history": [
            "in us history class",
            "for my history homework",
            "american history question about",
        ],
        "math": [
            "in my math class",
            "for algebra homework",
            "studying for a math quiz on",
        ],
        "english": [
            "in english class",
            "for my lit essay on",
            "english homework about",
        ],
        "art": [
            "in art class",
            "for my studio art project",
            "art history question about",
        ],
        "politics": [
            "in civics",
            "for government class",
            "politics question about",
        ],
        "biology": [
            "in bio lab",
            "for biology homework",
            "ap biology question on",
        ],
        "physics": [
            "in physics class",
            "for my physics homework",
            "studying mechanics and",
        ],
        "chemistry": [
            "in chemistry class",
            "for my chem lab",
            "ap chemistry question on",
        ],
        "technology": [
            "in my computer class",
            "for coding homework",
            "cs class question about",
        ],
    }
    rows = []
    for intent, topic_map in KNOWLEDGE_BASE.items():
        teasers = subject_teasers.get(intent, ["for school"])
        for keyword in topic_map.keys():
            if keyword == "default":
                continue
            cleaned = keyword.replace("_", " ").strip()
            prompts = [tpl.format(keyword=cleaned) for tpl in AUGMENTATION_TEMPLATES]
            for teaser in teasers[:2]:
                prompts.append(f"{teaser} {cleaned}")
            for q in prompts:
                rows.append({"question": q, "intent": intent})
    return pd.DataFrame(rows)


def build_model(num_classes: int) -> Sequential:
    """Conv1D stack — captures local word order (bigrams/trigrams) vs bag-of-words pooling."""
    return Sequential(
        [
            Embedding(
                input_dim=VOCAB_SIZE,
                output_dim=EMBEDDING_DIM,
                input_length=MAX_LEN,
                name="embedding",
            ),
            Conv1D(filters=128, kernel_size=3, activation="relu", padding="same", name="conv1_3"),
            Conv1D(filters=64, kernel_size=2, activation="relu", padding="same", name="conv1_2"),
            GlobalMaxPooling1D(name="pool"),
            BatchNormalization(name="bn"),
            Dense(128, activation="relu", name="dense1"),
            Dropout(0.4, name="drop1"),
            Dense(64, activation="relu", name="dense2"),
            Dropout(0.3, name="drop2"),
            Dense(num_classes, activation="softmax", name="softmax"),
        ],
        name="Academic_Tutor_Conv1D",
    )


def print_calibration_summary(
    model: tf.keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Binned confidence vs accuracy (ECE-style diagnostic from skill reference)."""
    probs = model.predict(X_test, verbose=0)
    predicted = np.argmax(probs, axis=1)
    confidence = np.max(probs, axis=1)
    correct = (predicted == y_test).astype(np.float64)

    print("\n=== Confidence calibration (max softmax) ===")
    print(f"{'Bin':<12} {'n':>6} {'avg conf':>10} {'accuracy':>10} {'gap':>8}")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_test)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (confidence >= lo) & (confidence < hi) if i < n_bins - 1 else (confidence >= lo) & (confidence <= hi)
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        acc_bin = float(correct[mask].mean())
        conf_bin = float(confidence[mask].mean())
        gap = acc_bin - conf_bin
        ece += (cnt / max(n, 1)) * abs(gap)
        flag = ""
        if gap < -0.1:
            flag = "  (overconfident)"
        elif gap > 0.1:
            flag = "  (underconfident)"
        print(f"{lo:.1f}-{hi:.1f}      {cnt:>6}  {conf_bin:>10.3f}  {acc_bin:>10.3f}  {gap:>+8.3f}{flag}")
    print(f"Weighted |gap| (ECE-like): {ece:.4f}\n")
    return float(ece)


SANITY_CHECK_CASES: Tuple[Tuple[str, str], ...] = (
    ("what is a noun", "english"),
    ("what is photosynthesis", "biology"),
    ("who was george washington", "american_history"),
    ("what is the pythagorean theorem", "math"),
    ("what is democracy", "politics"),
    ("what is impressionism", "art"),
    ("what is gravity", "physics"),
    ("how do you factor a quadratic", "math"),
    ("explain the civil war", "american_history"),
    ("what is mitosis", "biology"),
    ("what are metaphors", "english"),
    ("explain newtons laws", "physics"),
    ("what is the bill of rights", ("american_history", "politics")),
    ("what is an integer", "math"),
    ("what does a painting composition mean", "art"),
    ("what is a mole in chemistry", "chemistry"),
    ("what is an ip address", "technology"),
    ("what is tcp vs udp", "technology"),
)


def run_sanity_check(
    model: tf.keras.Model,
    tokenizer: Tokenizer,
    le: LabelEncoder,
    cases: Sequence[Tuple[str, Union[str, Tuple[str, ...]]]],
    max_len: int,
) -> float:
    """Fixed question suite — must align with label encoder classes."""
    valid = set(le.classes_)
    print("\n" + "=" * 60)
    print("  Sanity check - known questions")
    print("=" * 60)
    passed = 0
    skipped = 0
    for question, expected in cases:
        allowed = (expected,) if isinstance(expected, str) else tuple(expected)
        bad = [a for a in allowed if a not in valid]
        if bad:
            print(f"  [SKIP] expected label(s) {bad!r} not in encoder")
            skipped += 1
            continue
        seq = tokenizer.texts_to_sequences([question.lower()])
        padded = pad_sequences(seq, maxlen=max_len, padding="post", truncating="post")
        probs = model.predict(padded, verbose=0)[0]
        pred_idx = int(np.argmax(probs))
        predicted = le.classes_[pred_idx]
        conf = float(probs[pred_idx]) * 100.0
        ok = predicted in allowed
        if ok:
            passed += 1
        tag = "PASS" if ok else "FAIL"
        print(f"  [{tag}] [{conf:5.1f}%] {question!r}")
        if not ok:
            print(f"       expected one of {allowed!r} | got {predicted!r}")
    n = len(cases) - skipped
    rate = passed / n if n else 0.0
    print(f"\n  Result: {passed}/{n} passed ({rate * 100:.0f}%)")
    if rate < 0.9:
        print("  [WARNING] Below 90% - review data/architecture before relying on this run.")
    else:
        print("  [OK] Sanity suite at or above 90%.")
    return rate


# ── 1. Load Dataset ───────────────────────────────────────────────────────────
def main(argv: Iterable[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Train Academic Tutor subject classifier.")
    ap.add_argument(
        "--audit-only",
        action="store_true",
        help="Load data + augmentation, print audit, and exit (no training).",
    )
    args = ap.parse_args(list(argv) if argv is not None else None)

    print("=" * 60)
    print("  Academic Tutor Chatbot - Training")
    print("=" * 60)

    df = pd.read_csv(ROOT / "academic_dataset.csv")
    base_count = len(df)
    aug_df = build_augmented_examples()
    df = pd.concat([df, aug_df], ignore_index=True)
    df["question"] = df["question"].astype(str).str.strip()
    df["intent"] = df["intent"].astype(str).str.strip()
    df = df.drop_duplicates(subset=["question", "intent"]).reset_index(drop=True)

    label_errors = validate_dataset_labels(df)
    if label_errors:
        for e in label_errors:
            print(f"[ERROR] {e}")
        raise SystemExit(1)

    print_data_audit(df, MAX_LEN)

    if args.audit_only:
        print("[INFO] --audit-only set; skipping training.\n")
        return

    print(f"\n[INFO] Base dataset samples: {base_count}")
    print(f"[INFO] Added synthetic samples from knowledge base: {len(aug_df)}")
    print(f"[INFO] Final training samples: {len(df)}")
    print(f"[INFO] Subjects: {sorted(df['intent'].unique())}\n")

    questions = df["question"].values
    labels = df["intent"].values

    # ── 2. Encode Labels ──────────────────────────────────────────────────────
    le = LabelEncoder()
    encoded_labels = le.fit_transform(labels)
    num_classes = len(le.classes_)
    print(f"[INFO] Number of classes: {num_classes}")

    # ── 3. Tokenize Text ──────────────────────────────────────────────────────
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(questions)
    sequences = tokenizer.texts_to_sequences(questions)
    padded = pad_sequences(sequences, maxlen=MAX_LEN, padding="post", truncating="post")

    print(f"[INFO] Vocabulary cap (num_words): {VOCAB_SIZE}")
    print(f"[INFO] Word index size: {len(tokenizer.word_index)} unique tokens")
    print(f"[INFO] Padded sequence shape: {padded.shape}\n")

    # ── 4. Train / Validation / Test Split ────────────────────────────────────
    X_temp, X_test, y_temp, y_test = train_test_split(
        padded, encoded_labels, test_size=TEST_SIZE, random_state=SEED, stratify=encoded_labels
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=VAL_SIZE, random_state=SEED, stratify=y_temp
    )

    print(f"[INFO] Train samples:      {len(X_train)}")
    print(f"[INFO] Validation samples: {len(X_val)}")
    print(f"[INFO] Test samples:       {len(X_test)}\n")

    uniq_train = np.unique(y_train)
    cw_arr = compute_class_weight(class_weight="balanced", classes=uniq_train, y=y_train)
    class_weight_dict = {int(c): float(w) for c, w in zip(uniq_train, cw_arr)}
    print("[INFO] Using balanced class_weight for imbalanced subject counts.\n")

    # ── 5. Build Model ────────────────────────────────────────────────────────
    model = build_model(num_classes)
    optimizer = Adam(
        learning_rate=LEARNING_RATE,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
        clipnorm=1.0,
    )
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    callbacks = [
        EarlyStopping(
            monitor="val_accuracy",
            patience=EARLY_PATIENCE,
            restore_best_weights=True,
            verbose=1,
            min_delta=0.001,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=LR_PATIENCE,
            min_lr=1e-6,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=str(CHECKPOINT_DIR / "best_model.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=0,
        ),
    ]

    # ── 6. Train ───────────────────────────────────────────────────────────────
    print("\n[INFO] Training started...\n")
    history = model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1,
    )

    # ── 7. Evaluate on Test Set ───────────────────────────────────────────────
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n{'=' * 60}")
    print(f"  Final Test Accuracy:  {test_acc * 100:.2f}%")
    print(f"  Final Test Loss:      {test_loss:.4f}")
    print(f"{'=' * 60}\n")

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    print("=" * 60)
    print("  Per-class classification report (test)")
    print("=" * 60)
    print(classification_report(y_test, y_pred, target_names=le.classes_, digits=3))
    report_dict = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
    per_class_f1 = [(k, v["f1-score"]) for k, v in report_dict.items() if k in le.classes_]
    if per_class_f1:
        worst = min(per_class_f1, key=lambda x: x[1])
        print(f"[INFO] Lowest F1 subject: {worst[0]} ({worst[1]:.3f})\n")

    print_calibration_summary(model, X_test, y_test)
    run_sanity_check(model, tokenizer, le, SANITY_CHECK_CASES, MAX_LEN)

    # ── 8. Plot Training vs Validation Accuracy ───────────────────────────────
    epochs_ran = range(1, len(history.history["accuracy"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Academic Tutor Chatbot — Training Results", fontsize=14, fontweight="bold")

    axes[0].plot(epochs_ran, history.history["accuracy"], label="Train Accuracy", color="steelblue")
    axes[0].plot(epochs_ran, history.history["val_accuracy"], label="Validation Accuracy", color="darkorange")
    axes[0].set_title("Training vs Validation Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs_ran, history.history["loss"], label="Train Loss", color="steelblue")
    axes[1].plot(epochs_ran, history.history["val_loss"], label="Validation Loss", color="darkorange")
    axes[1].set_title("Training vs Validation Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(ROOT / "training_results.png", dpi=150, bbox_inches="tight")
    print(f"[INFO] Training plot saved -> {ROOT / 'training_results.png'}")

    # ── 9. Confusion Matrix ───────────────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot(ax=ax2, xticks_rotation=45, colorbar=False, cmap="Blues")
    ax2.set_title("Confusion Matrix — Test Set", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(ROOT / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    print(f"[INFO] Confusion matrix saved -> {ROOT / 'confusion_matrix.png'}\n")

    # ── 10. Save Artifacts ────────────────────────────────────────────────────
    model.save(ROOT / "academic_chatbot_model.keras")
    print(f"[INFO] Model saved -> {ROOT / 'academic_chatbot_model.keras'}")

    with open(ROOT / "tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)
    print(f"[INFO] Tokenizer saved -> {ROOT / 'tokenizer.pkl'}")

    with open(ROOT / "label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)
    print(f"[INFO] Label encoder saved -> {ROOT / 'label_encoder.pkl'}")

    print("\n[DONE] Training complete. Run chatbot.py to start the tutor!\n")


if __name__ == "__main__":
    main()
