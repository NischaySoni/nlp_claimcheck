"""
Baseline 1: SVM + TF-IDF for Binary Claim Check-Worthiness Detection
=====================================================================
Uses TF-IDF features from the 'claim' column and trains a LinearSVC
classifier for binary check-worthiness (check-worthy vs non-check-worthy).
"""

import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# ─── Paths ────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
TRAIN_PATH = os.path.join(DATA_DIR, "train_data.csv")
VAL_PATH = os.path.join(DATA_DIR, "val_data.csv")
TEST_PATH = os.path.join(DATA_DIR, "test_data.csv")

# ─── Load Data ────────────────────────────────────────────────────────
print("=" * 60)
print("Baseline 1: SVM + TF-IDF")
print("=" * 60)

train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)
test_df = pd.read_csv(TEST_PATH)

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# Use the 'claim' column as input text, 'bin_label' as the target
X_train = train_df["claim"].fillna("").astype(str)
y_train = train_df["bin_label"].values

X_val = val_df["claim"].fillna("").astype(str)
y_val = val_df["bin_label"].values

X_test = test_df["claim"].fillna("").astype(str)
y_test = test_df["bin_label"].values

# ─── TF-IDF Vectorization ────────────────────────────────────────────
print("\nFitting TF-IDF vectorizer ...")
tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=2,
)
X_train_tfidf = tfidf.fit_transform(X_train)
X_val_tfidf = tfidf.transform(X_val)
X_test_tfidf = tfidf.transform(X_test)

# ─── Train SVM ────────────────────────────────────────────────────────
print("Training LinearSVC ...")
svm = LinearSVC(
    class_weight="balanced",
    max_iter=10000,
    C=1.0,
    random_state=42,
)
svm.fit(X_train_tfidf, y_train)

# ─── Evaluate ─────────────────────────────────────────────────────────
def evaluate(name, X, y_true):
    y_pred = svm.predict(X)
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    cw_f1 = f1_score(y_true, y_pred, pos_label=1)
    ncw_f1 = f1_score(y_true, y_pred, pos_label=0)

    print(f"\n{'─' * 40}")
    print(f"  {name} Results")
    print(f"{'─' * 40}")
    print(f"  Accuracy       : {acc:.4f}")
    print(f"  Macro F1       : {macro_f1:.4f}")
    print(f"  CW F1 (cls=1)  : {cw_f1:.4f}")
    print(f"  NCW F1 (cls=0) : {ncw_f1:.4f}")
    print(f"\n  Classification Report:\n")
    print(
        classification_report(
            y_true, y_pred, target_names=["Non-Check-Worthy", "Check-Worthy"]
        )
    )
    return acc, macro_f1, cw_f1, ncw_f1


val_metrics = evaluate("Validation", X_val_tfidf, y_val)
test_metrics = evaluate("Test", X_test_tfidf, y_test)

# ─── Summary Table ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  Summary: SVM + TF-IDF Baseline")
print("=" * 60)
print(f"  {'Split':<12} {'Acc':>8} {'m-F1':>8} {'cw-F1':>8} {'ncw-F1':>8}")
print(f"  {'─'*44}")
print(
    f"  {'Validation':<12} {val_metrics[0]:>8.4f} {val_metrics[1]:>8.4f} {val_metrics[2]:>8.4f} {val_metrics[3]:>8.4f}"
)
print(
    f"  {'Test':<12} {test_metrics[0]:>8.4f} {test_metrics[1]:>8.4f} {test_metrics[2]:>8.4f} {test_metrics[3]:>8.4f}"
)
print("=" * 60)
