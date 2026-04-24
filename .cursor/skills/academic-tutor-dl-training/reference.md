# Reference — Academic Tutor DL training

Deep cuts: audit scripts, model snippets, checklists. Read sections as needed.

---

## Architecture audit

### Baseline stack (typical `train.py`)

```
Embedding(VOCAB_SIZE=3000, EMBEDDING_DIM=64)
→ GlobalAveragePooling1D()
→ Dense(128, relu) → Dropout(0.4)
→ Dense(64, relu)  → Dropout(0.3)
→ Dense(num_classes, softmax)
```

### Known weaknesses

| Problem | Root cause | Impact |
|---------|------------|--------|
| Word order ignored | `GlobalAveragePooling1D` averages all positions | "What is a noun" vs "A noun is what" look identical |
| Vocabulary too small | `VOCAB_SIZE=3000` with OOV | Unseen phrasings map to `<OOV>` |
| `MAX_LEN=20` clips | Truncation | Long questions lose tail |
| Random embeddings | Not pretrained | Needs more data to converge |
| No positional structure | Bag-of-tokens | Weak on negation, qualifiers, structure |
| Single softmax head | No calibration | Raw scores are not reliable confidence % |

### Why more data alone may not help

The model may learn co-occurrence statistics, not robust language understanding. Architecture limits persist regardless of row count.

### Architecture trade-offs (upgrade options)

| Option | Accuracy | Training speed | Deployment size | Complexity |
|--------|----------|----------------|------------------|--------------|
| A: Conv1D | Good | Fast | Small (~2 MB) | Low |
| B: BiLSTM | Better | Medium | Small (~5 MB) | Medium |
| C: USE (TF Hub) | Best | Slow first run | Large (~1 GB) | Medium |

**Suggested path:** Try A, move to B if validation accuracy stays below ~85%, use C when quality outweighs model size.

---

## Data audit

Run before retraining:

```python
import pandas as pd
from collections import Counter

df = pd.read_csv("academic_dataset.csv")

# Check class balance
print("=== Class Distribution ===")
print(df['intent'].value_counts())
print(f"\nSmallest class: {df['intent'].value_counts().min()} samples")
print(f"Largest class:  {df['intent'].value_counts().max()} samples")
print(f"Imbalance ratio: {df['intent'].value_counts().max() / df['intent'].value_counts().min():.1f}x")

# Check for duplicates
dupes = df[df.duplicated(subset=['question'], keep=False)]
print(f"\n=== Duplicates: {len(dupes)} rows ===")
print(dupes.head(10))

# Check question length distribution
df['length'] = df['question'].str.split().str.len()
print(f"\n=== Question Lengths ===")
print(df['length'].describe())
print(f"Questions longer than MAX_LEN=20: {(df['length'] > 20).sum()}")

# Check for label noise — short suspicious questions
print("\n=== Very short questions (<=2 words) ===")
print(df[df['length'] <= 2][['question', 'intent']].head(20))
```

**Fix rules:**

- Imbalance ratio > 3x: upsample minority or downsample majority
- Duplicates: remove — they inflate accuracy without generalization
- Label noise: fix manually — garbage in, garbage out

### Row validation helper

```python
def validate_row(question: str, intent: str) -> list[str]:
    errors = []

    VALID_INTENTS = {
        "american_history", "math", "english",
        "art", "politics", "biology", "physics", "out_of_scope"
    }

    if intent not in VALID_INTENTS:
        errors.append(f"Unknown intent: '{intent}'")

    words = question.strip().split()
    if len(words) < 2:
        errors.append("Question too short (< 2 words)")
    if len(words) > 40:
        errors.append("Question too long (> 40 words) — consider trimming")
    if not any(c.isalpha() for c in question):
        errors.append("Question contains no alphabetic characters")
    if question == question.upper() and len(question) > 3:
        errors.append("Question is ALL CAPS — likely a data error")

    return errors
```

---

## Augmentation templates

Add patterns that match real student language to `build_augmented_examples()` (or equivalent):

```python
AUGMENTATION_TEMPLATES = [
    # Baseline templates (keep as needed)
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

    # Question-word variety
    "how does {keyword} work",
    "why is {keyword} important",
    "when did {keyword} happen",
    "who invented {keyword}",
    "what are the parts of {keyword}",
    "give me an example of {keyword}",
    "define {keyword}",
    "what causes {keyword}",
    "describe {keyword}",
    "summarize {keyword}",

    # Exam / homework phrasing
    "study guide for {keyword}",
    "exam question about {keyword}",
    "notes on {keyword}",
    "what should i know about {keyword}",
    "is {keyword} on the test",

    # Confusion / frustration
    "i am confused about {keyword}",
    "struggling with {keyword}",
    "i keep getting {keyword} wrong",
    "can you re-explain {keyword}",
    "what am i missing about {keyword}",

    # Comparison
    "difference between {keyword} and",
    "compare {keyword} to",
    "how is {keyword} different from",

    # Contextual
    "in the context of {keyword}",
    "regarding {keyword}",
    "relating to {keyword}",
]
```

---

## Architecture options

### Option A — Conv1D (minimal change)

Replace global average pooling with local n-gram features:

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding, Conv1D, GlobalMaxPooling1D,
    Dense, Dropout, BatchNormalization
)

model = Sequential([
    Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM, input_length=MAX_LEN),

    Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'),
    Conv1D(filters=64,  kernel_size=2, activation='relu', padding='same'),
    GlobalMaxPooling1D(),

    BatchNormalization(),
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
], name="Academic_Tutor_Conv1D")
```

Suggested constants when moving to Conv1D:

```python
VOCAB_SIZE    = 5000
MAX_LEN       = 30
EMBEDDING_DIM = 128
BATCH_SIZE    = 32
```

### Option B — Bidirectional LSTM

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding, Bidirectional, LSTM,
    Dense, Dropout, BatchNormalization, SpatialDropout1D
)

model = Sequential([
    Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM,
                input_length=MAX_LEN, mask_zero=True),
    SpatialDropout1D(0.2),

    Bidirectional(LSTM(64, return_sequences=True, dropout=0.2)),
    Bidirectional(LSTM(32, dropout=0.2)),

    BatchNormalization(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
], name="Academic_Tutor_BiLSTM")
```

Suggested constants for BiLSTM:

```python
VOCAB_SIZE    = 6000
MAX_LEN       = 35
EMBEDDING_DIM = 128
EPOCHS        = 80
BATCH_SIZE    = 32
LEARNING_RATE = 0.001
```

### Option C — Universal Sentence Encoder (TensorFlow Hub)

Sentence-level encoding; pass raw strings (no tokenization for the hub layer):

```python
import tensorflow_hub as hub
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
import tensorflow as tf

EMBED_URL = "https://tfhub.dev/google/universal-sentence-encoder/4"

def build_use_model(num_classes: int):
    text_input = tf.keras.Input(shape=(), dtype=tf.string, name="text_input")

    embed_layer = hub.KerasLayer(
        EMBED_URL, input_shape=[], dtype=tf.string, trainable=False,
        name="universal_sentence_encoder"
    )
    embeddings = embed_layer(text_input)

    x = Dense(256, activation='relu')(embeddings)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    output = Dense(num_classes, activation='softmax')(x)

    return tf.keras.Model(inputs=text_input, outputs=output, name="Academic_Tutor_USE")

# X_train = raw string array; model.fit(X_train, y_train, ...)
```

---

## Training

### Optimizer

```python
from tensorflow.keras.optimizers import Adam

optimizer = Adam(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-7,
    clipnorm=1.0
)

model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

### Callbacks

```python
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)

callbacks = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=15,
        restore_best_weights=True,
        verbose=1,
        min_delta=0.001
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=6,
        min_lr=1e-6,
        verbose=1
    ),
    ModelCheckpoint(
        filepath=str(ROOT / "checkpoints" / "best_model.keras"),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=0
    ),
]
```

Adjust `ROOT` / path to match your project layout.

### Class weights (imbalanced intents)

```python
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(enumerate(class_weights))

history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val, y_val),
    class_weight=class_weight_dict,
    callbacks=callbacks,
    verbose=1
)
```

### Label smoothing

Requires one-hot labels and `CategoricalCrossentropy`:

```python
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import CategoricalCrossentropy

y_train_oh = to_categorical(y_train, num_classes)
y_val_oh   = to_categorical(y_val,   num_classes)
y_test_oh  = to_categorical(y_test,  num_classes)

model.compile(
    optimizer=optimizer,
    loss=CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)
```

---

## Evaluation

### Per-class report

```python
from sklearn.metrics import classification_report

y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

print("\n" + "=" * 70)
print("  Per-Class Classification Report")
print("=" * 70)
print(classification_report(
    y_test, y_pred,
    target_names=le.classes_,
    digits=3
))

report_dict = classification_report(
    y_test, y_pred,
    target_names=le.classes_,
    output_dict=True
)
worst_class = min(
    [(k, v['f1-score']) for k, v in report_dict.items()
     if k in le.classes_],
    key=lambda x: x[1]
)
print(f"\n[WARNING] Worst performing subject: {worst_class[0]} (F1={worst_class[1]:.3f})")
```

### Confidence calibration (binned)

```python
def check_confidence_calibration(model, X_test, y_test, n_bins=10):
    probs = model.predict(X_test, verbose=0)
    predicted = np.argmax(probs, axis=1)
    confidence = np.max(probs, axis=1)
    correct = (predicted == y_test).astype(int)

    print("\n=== Confidence Calibration ===")
    print(f"{'Conf. Range':<15} {'Samples':>8} {'Avg Conf.':>10} {'Actual Acc.':>12} {'Gap':>8}")
    print("-" * 55)

    bins = np.linspace(0, 1, n_bins + 1)
    for i in range(n_bins):
        mask = (confidence >= bins[i]) & (confidence < bins[i + 1])
        if mask.sum() == 0:
            continue
        avg_conf = confidence[mask].mean()
        avg_acc = correct[mask].mean()
        gap = avg_acc - avg_conf
        flag = " ← OVERCONFIDENT" if gap < -0.1 else (" ← UNDERCONFIDENT" if gap > 0.1 else "")
        print(
            f"{bins[i]:.1f}-{bins[i + 1]:.1f}        {mask.sum():>8}  {avg_conf:>9.3f}  {avg_acc:>11.3f}  {gap:>+7.3f}{flag}"
        )

    ece = sum(
        (mask.sum() / len(y_test)) * abs(confidence[mask].mean() - correct[mask].mean())
        for i in range(n_bins)
        for mask in [(confidence >= bins[i]) & (confidence < bins[i + 1])]
        if mask.sum() > 0
    )
    print(f"\nExpected Calibration Error (ECE): {ece:.4f}")
    print("(Below 0.05 is good. Above 0.15 means confidence scores are unreliable.)")


check_confidence_calibration(model, X_test, y_test)
```

### Sanity check suite (tokenized models)

```python
SANITY_CHECK_CASES = [
    ("what is a noun",                  "english"),
    ("what is photosynthesis",          "biology"),
    ("who was george washington",       "american_history"),
    ("what is the pythagorean theorem", "math"),
    ("what is democracy",               "politics"),
    ("what is impressionism",           "art"),
    ("what is gravity",                 "physics"),
    ("how do you factor a quadratic",   "math"),
    ("explain the civil war",           "american_history"),
    ("what is mitosis",                 "biology"),
    ("what are metaphors",              "english"),
    ("explain newtons laws",            "physics"),
    ("what is the bill of rights",      "american_history"),
    ("what is an integer",              "math"),
    ("what does a painting composition mean", "art"),
    ("whats the best pizza topping",    "out_of_scope"),
    ("who won the super bowl",          "out_of_scope"),
]


def run_sanity_check(model, tokenizer, le, test_cases, max_len):
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    print("\n" + "=" * 70)
    print("  Sanity Check — Known Questions")
    print("=" * 70)
    passed = 0
    for question, expected in test_cases:
        seq = tokenizer.texts_to_sequences([question])
        padded = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')
        probs = model.predict(padded, verbose=0)[0]
        pred_idx = int(np.argmax(probs))
        predicted = le.classes_[pred_idx]
        confidence = probs[pred_idx] * 100
        ok = "PASS" if predicted == expected else "FAIL"
        if predicted == expected:
            passed += 1
        print(f"  [{ok}] [{confidence:5.1f}%] '{question}'")
        if predicted != expected:
            print(f"       Expected: {expected} | Got: {predicted}")

    rate = passed / len(test_cases)
    print(f"\n  Result: {passed}/{len(test_cases)} passed ({rate * 100:.0f}%)")
    if rate < 0.9:
        print("  [FAIL] Below 90% — do NOT deploy this model.")
    else:
        print("  [PASS] Ready for integration.")
    return rate


sanity_score = run_sanity_check(model, tokenizer, le, SANITY_CHECK_CASES, MAX_LEN)
```

For USE / string-input models, adapt the sanity loop to pass raw strings instead of `pad_sequences`.

---

## Hyperparameters

Example top-of-file block when using upgraded architectures:

```python
# Architecture: "conv1d" | "bilstm" | "use"
ARCHITECTURE   = "conv1d"

VOCAB_SIZE     = 5000
MAX_LEN        = 30
EMBEDDING_DIM  = 128
EPOCHS         = 100
BATCH_SIZE     = 32
TEST_SIZE      = 0.15
VAL_SIZE       = 0.15
LEARNING_RATE  = 0.001
LABEL_SMOOTHING = 0.05
DROPOUT_1      = 0.4
DROPOUT_2      = 0.3
MIN_SAMPLES_PER_CLASS = 30
```

Wire `ARCHITECTURE` / smoothing into your actual `train.py` branches as implemented.

---

## Production checklist

**Model quality**

- [ ] Sanity check passes >= 90% on the fixed question set
- [ ] Per-class F1 >= 0.75 for every subject (tune threshold to product needs)
- [ ] ECE < 0.10 (if you rely on confidence)
- [ ] Val accuracy within ~5% of train (watch overfitting)

**Data health**

- [ ] No class below `MIN_SAMPLES_PER_CLASS` (e.g. 30)
- [ ] Imbalance ratio < 3x
- [ ] No duplicate questions in training split
- [ ] Label encoder classes match what `chatbot.py` expects

**Inference integration**

- [ ] `tokenizer.pkl` behavior matches training preprocessing
- [ ] `LabelEncoder.pkl` classes align with `KNOWLEDGE_BASE` / intent keys
- [ ] Out-of-scope threshold tunable at inference (not only hardcoded)
- [ ] Confidence display documented (normalized vs raw softmax if applicable)

**Artifacts**

- [ ] `academic_chatbot_model.keras` (or project name)
- [ ] `tokenizer.pkl`, `label_encoder.pkl`
- [ ] `training_results.png`, `confusion_matrix.png` (optional but useful)
- [ ] Log sanity check output for the run

---

## Common errors

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| "What is a noun?" → out_of_scope | Pooling destroys structure | Conv1D or BiLSTM |
| ~46% confidence on easy questions | Overfit / miscalibration | Augmentation, label smoothing, calibration check |
| Val accuracy oscillates | LR too high | `ReduceLROnPlateau`, lower base LR |
| Train 98% / val 70% | Overfitting | Dropout, augmentation, smaller model |
| One subject always wrong | Few or bad labels for that class | Fix data; class weights |
| `KeyError` at inference | Encoder vs app mismatch | Compare `le.classes_` to app intents |
| OOM | Batch too large | Reduce `BATCH_SIZE` |
