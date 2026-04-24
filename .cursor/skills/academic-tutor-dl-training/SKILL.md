---
name: academic-tutor-dl-training
description: >-
  Guides diagnosis and improvement of the Keras/TensorFlow intent-classification
  training pipeline for the Academic Tutor chatbot (train.py, academic_dataset.csv).
  Covers data quality, augmentation templates, Conv1D/BiLSTM/USE architectures,
  optimizers and callbacks, class weights, label smoothing, metrics, calibration,
  sanity checks, and production readiness. Use when improving training, fixing
  accuracy, retraining, tuning hyperparameters, evaluating NLU model health, or
  working on build_augmented_examples / expand_academic_dataset for this project.
---

# Academic Tutor — deep learning training

## When to use

Apply when editing `train.py`, dataset expansion, or training/eval artifacts for this chatbot. Typical triggers: improve training, fix accuracy, retrain, hyperparameter tuning, misclassification, or confidence/calibration issues.

## Operating principles

1. **Data before architecture** — More rows alone do not fix bag-of-words limits; audit labels, balance, duplicates, and length first.
2. **Match inference** — Label encoder classes and tokenizer must align with `chatbot.py` and knowledge base intent keys.
3. **Read `train.py` first** — Confirm the current stack and constants (`VOCAB_SIZE`, `MAX_LEN`, layers) before recommending changes.

## Known limitations (baseline stack)

The common baseline uses embedding + `GlobalAveragePooling1D`, which discards word order: negation, question structure, and paraphrases can look identical. Small vocabulary, short `MAX_LEN`, and non-pretrained embeddings increase OOV and truncation. Raw softmax is not a calibrated confidence score.

For the architecture diagram, weakness table, and trade-offs between Conv1D / BiLSTM / USE, see [reference.md — Architecture audit](reference.md#architecture-audit).

## Phased workflow

Use this checklist and pull code from reference only for the step you need.

- [ ] **Phase 1 — Data**: Class balance, duplicates, length vs `MAX_LEN`, label noise. Run the audit snippet in [reference.md — Data audit](reference.md#data-audit). Apply validation rules and imbalance/duplicate fixes before retraining.
- [ ] **Phase 2 — Architecture**: Prefer incremental change: Conv1D + `GlobalMaxPooling1D`, then BiLSTM if accuracy still weak, then USE if quality over size. Code in [reference.md — Architecture options](reference.md#architecture-options).
- [ ] **Phase 3 — Training**: Explicit `Adam`, `EarlyStopping`, `ReduceLROnPlateau`, `ModelCheckpoint`; optional `class_weight`; label smoothing with categorical loss if needed. See [reference.md — Training](reference.md#training).
- [ ] **Phase 4 — Evaluation**: Per-class report, calibration / ECE, sanity question suite. See [reference.md — Evaluation](reference.md#evaluation).
- [ ] **Phase 5 — Hyperparameters**: Suggested constants block for upgraded stacks in [reference.md — Hyperparameters](reference.md#hyperparameters).
- [ ] **Phase 6 — Production**: Gate deployment with the checklist in [reference.md — Production checklist](reference.md#production-checklist).

## Quick symptom to first action

| Symptom | First action |
|---------|----------------|
| Easy questions wrong or marked out-of-scope | Upgrade architecture (order-aware) and audit data for that intent |
| Low confidence on easy real phrasing | More augmentation variety; check calibration; consider label smoothing |
| Val metric oscillates | Add `ReduceLROnPlateau`; reduce learning rate |
| Train much higher than val | Stronger dropout, more diverse augmentation, or smaller capacity |
| One subject always wrong | `classification_report` for that class; fix labels and sample count |
| `KeyError` or wrong route at inference | Compare `le.classes_` to knowledge base keys exactly |
| OOM while training | Lower `BATCH_SIZE` (e.g. 16 or 8) |

Extended symptom table: [reference.md — Common errors](reference.md#common-errors).

## Augmentation

Extend `build_augmented_examples()` (or dataset expansion) with student-realistic patterns: wh-questions, exam/study phrasing, confusion/frustration, comparisons, short context phrases. Full `AUGMENTATION_TEMPLATES` list: [reference.md — Augmentation templates](reference.md#augmentation-templates).

## Installing this skill elsewhere

- **This repo**: Cursor loads `.cursor/skills/` automatically.
- **All projects (personal)**: Copy the folder `academic-tutor-dl-training` to `~/.cursor/skills/` and adjust file names if your other projects use different paths.
