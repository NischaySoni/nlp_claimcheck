"""
Baseline: Fine-tuned RoBERTa
==============================
Matches the paper's experimental setup exactly:

  Setup 1 — Binary check-worthiness:
    One RoBERTa model fine-tuned for binary CW classification.

  Setup 2 — Rationality labels:
    Six INDEPENDENT RoBERTa models, one per rationality label,
    each fine-tuned as a binary classifier on the CW-only subset.
    This matches the paper: "six rationality labels (R1-R6) are
    deemed independent tasks" (Section V-A).

Usage:
  python baseline_roberta.py
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import (
    RobertaForSequenceClassification,
    RobertaTokenizerFast,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from torch.utils.data import Dataset

# ─── Paths ────────────────────────────────────────────────────────────
DATA_DIR   = os.path.join(os.path.dirname(__file__), "..", "data")
TRAIN_PATH = os.path.join(DATA_DIR, "train_data.csv")
VAL_PATH   = os.path.join(DATA_DIR, "val_data.csv")
TEST_PATH  = os.path.join(DATA_DIR, "test_data.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "roberta_baseline_output")

# ─── Rationality Columns ──────────────────────────────────────────────
RATIONALITY_COLS = [
    "verifiable_factual_claim",
    "false_info",
    "general_public_interest",
    "harmful",
    "fact_checker_interest",
    "govt_interest",
]
NUM_RAT = len(RATIONALITY_COLS)

# ─── Hyperparameters ──────────────────────────────────────────────────
MODEL_NAME    = "roberta-base"
MAX_LEN       = 128
BATCH_SIZE    = 8
LEARNING_RATE = 2e-5
EPOCHS        = 5
PATIENCE      = 2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
print("=" * 60)
print("  Baseline: Fine-tuned RoBERTa")
print("=" * 60)

# ─── Load Data ────────────────────────────────────────────────────────
train_df = pd.read_csv(TRAIN_PATH)
val_df   = pd.read_csv(VAL_PATH)
test_df  = pd.read_csv(TEST_PATH)
print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_NAME)


# ─── Dataset ──────────────────────────────────────────────────────────
class ClaimDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=MAX_LEN):
        self.texts     = [str(t) for t in texts]
        self.labels    = labels.tolist()
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ─── Training args ────────────────────────────────────────────────────
def make_training_args(output_subdir):
    return TrainingArguments(
        output_dir=os.path.join(OUTPUT_DIR, output_subdir),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=50,
        save_total_limit=1,
        report_to="none",
        fp16=torch.cuda.is_available(),
    )


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
    }


# ══════════════════════════════════════════════════════════════════════
# SETUP 1 — Binary Check-Worthiness
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  SETUP 1: Binary Check-Worthiness")
print("=" * 60)

train_bin_ds = ClaimDataset(
    train_df["claim"].fillna(""), train_df["bin_label"].values, tokenizer)
val_bin_ds   = ClaimDataset(
    val_df["claim"].fillna(""),   val_df["bin_label"].values,   tokenizer)
test_bin_ds  = ClaimDataset(
    test_df["claim"].fillna(""),  test_df["bin_label"].values,  tokenizer)

bin_model = RobertaForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

bin_trainer = Trainer(
    model=bin_model,
    args=make_training_args("binary"),
    train_dataset=train_bin_ds,
    eval_dataset=val_bin_ds,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=PATIENCE)],
)
print("Training binary model ...")
bin_trainer.train()


def get_binary_metrics(trainer, dataset, y_true, split_name):
    out    = trainer.predict(dataset)
    preds  = np.argmax(out.predictions, axis=-1)
    acc    = accuracy_score(y_true, preds)
    mf1    = f1_score(y_true, preds, average="macro")
    cw_f1  = f1_score(y_true, preds, pos_label=1)
    ncw_f1 = f1_score(y_true, preds, pos_label=0)
    print(f"\n{'─'*50}")
    print(f"  {split_name} — Binary Check-Worthiness")
    print(f"{'─'*50}")
    print(f"  Accuracy       : {acc:.4f}")
    print(f"  Macro F1       : {mf1:.4f}")
    print(f"  CW F1 (cls=1)  : {cw_f1:.4f}")
    print(f"  NCW F1 (cls=0) : {ncw_f1:.4f}")
    print(classification_report(y_true, preds,
          target_names=["Non-Check-Worthy", "Check-Worthy"]))
    return acc, mf1, cw_f1, ncw_f1


val_bin  = get_binary_metrics(bin_trainer, val_bin_ds,  val_df["bin_label"].values,  "Validation")
test_bin = get_binary_metrics(bin_trainer, test_bin_ds, test_df["bin_label"].values, "Test")


# ══════════════════════════════════════════════════════════════════════
# SETUP 2 — Six Independent Rationality Label Models
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  SETUP 2: Rationality Labels (6 independent models)")
print("=" * 60)

train_cw = train_df[train_df["bin_label"] == 1].reset_index(drop=True)
val_cw   = val_df[val_df["bin_label"] == 1].reset_index(drop=True)
test_cw  = test_df[test_df["bin_label"] == 1].reset_index(drop=True)

print(f"CW subset — Train: {len(train_cw)} | Val: {len(val_cw)} | Test: {len(test_cw)}")

val_rat_f1s  = []
test_rat_f1s = []

for i, col in enumerate(RATIONALITY_COLS):
    print(f"\n  [{i+1}/6] Training model for: {col}")

    y_train = train_cw[col].fillna(0).astype(int).values
    y_val   = val_cw[col].fillna(0).astype(int).values
    y_test  = test_cw[col].fillna(0).astype(int).values

    tr_ds  = ClaimDataset(train_cw["claim"].fillna(""), y_train, tokenizer)
    v_ds   = ClaimDataset(val_cw["claim"].fillna(""),   y_val,   tokenizer)
    te_ds  = ClaimDataset(test_cw["claim"].fillna(""),  y_test,  tokenizer)

    rat_model = RobertaForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2)

    rat_trainer = Trainer(
        model=rat_model,
        args=make_training_args(f"rat_{i}_{col[:10]}"),
        train_dataset=tr_ds,
        eval_dataset=v_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=PATIENCE)],
    )
    rat_trainer.train()

    v_out    = rat_trainer.predict(v_ds)
    v_preds  = np.argmax(v_out.predictions, axis=-1)
    v_f1     = f1_score(y_val, v_preds, average="macro", zero_division=0)
    val_rat_f1s.append(v_f1)

    te_out   = rat_trainer.predict(te_ds)
    te_preds = np.argmax(te_out.predictions, axis=-1)
    te_f1    = f1_score(y_test, te_preds, average="macro", zero_division=0)
    test_rat_f1s.append(te_f1)

    print(f"    Val  macro F1: {v_f1:.4f}")
    print(f"    Test macro F1: {te_f1:.4f}")

# ─── Rationality Results ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("  Rationality Label Results")
print("=" * 60)
print(f"  {'Label':<35s}  {'Val F1':>8}  {'Test F1':>8}")
print(f"  {'─' * 55}")
for col, vf1, tf1 in zip(RATIONALITY_COLS, val_rat_f1s, test_rat_f1s):
    print(f"  {col:<35s}: {vf1:.4f}    {tf1:.4f}")

# ─── Summary Table ────────────────────────────────────────────────────
print("\n" + "=" * 95)
print("  Summary: RoBERTa Baseline")
print("=" * 95)
print(f"  {'Split':<12} {'Acc':>6} {'m-F1':>6} {'cw-F1':>7} {'ncw-F1':>8}  "
      + "  ".join(f"R{i+1}" for i in range(NUM_RAT)))
print(f"  {'─' * 88}")

val_r  = "  ".join(f"{v:.4f}" for v in val_rat_f1s)
test_r = "  ".join(f"{v:.4f}" for v in test_rat_f1s)
print(f"  {'Validation':<12} {val_bin[0]:>6.4f} {val_bin[1]:>6.4f} "
      f"{val_bin[2]:>7.4f} {val_bin[3]:>8.4f}  {val_r}")
print(f"  {'Test':<12} {test_bin[0]:>6.4f} {test_bin[1]:>6.4f} "
      f"{test_bin[2]:>7.4f} {test_bin[3]:>8.4f}  {test_r}")
print("=" * 95)
