"""
Baseline 2: Fine-tuned BERT for Binary Claim Check-Worthiness Detection
========================================================================
Fine-tunes bert-base-uncased on the 'claim' column for binary
check-worthiness classification using HuggingFace Trainer API.
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report

from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from torch.utils.data import Dataset

# ─── Paths ────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
TRAIN_PATH = os.path.join(DATA_DIR, "train_data.csv")
VAL_PATH = os.path.join(DATA_DIR, "val_data.csv")
TEST_PATH = os.path.join(DATA_DIR, "test_data.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "bert_baseline_output")

# ─── Hyperparameters ──────────────────────────────────────────────────
MODEL_NAME = "bert-base-uncased"
MAX_LEN = 128
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
EPOCHS = 5
PATIENCE = 2

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ─── Load Data ────────────────────────────────────────────────────────
print("=" * 60)
print("Baseline 2: Fine-tuned BERT")
print("=" * 60)

train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)
test_df = pd.read_csv(TEST_PATH)

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# ─── Tokenizer ────────────────────────────────────────────────────────
tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)


# ─── Dataset Class ────────────────────────────────────────────────────
class ClaimDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts.tolist()
        self.labels = labels.tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# Create datasets
train_dataset = ClaimDataset(
    train_df["claim"].fillna(""), train_df["bin_label"].values, tokenizer, MAX_LEN
)
val_dataset = ClaimDataset(
    val_df["claim"].fillna(""), val_df["bin_label"].values, tokenizer, MAX_LEN
)
test_dataset = ClaimDataset(
    test_df["claim"].fillna(""), test_df["bin_label"].values, tokenizer, MAX_LEN
)


# ─── Metrics ──────────────────────────────────────────────────────────
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "macro_f1": macro_f1}


# ─── Model ────────────────────────────────────────────────────────────
print("\nLoading BERT model ...")
model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=2
)

# ─── Training ─────────────────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    greater_is_better=True,
    logging_steps=50,
    save_total_limit=2,
    report_to="none",
    fp16=torch.cuda.is_available(),
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=PATIENCE)],
)

print("Starting training ...")
trainer.train()

# ─── Evaluate ─────────────────────────────────────────────────────────
def evaluate_split(name, dataset, y_true):
    preds_out = trainer.predict(dataset)
    y_pred = np.argmax(preds_out.predictions, axis=-1)

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


val_metrics = evaluate_split("Validation", val_dataset, val_df["bin_label"].values)
test_metrics = evaluate_split("Test", test_dataset, test_df["bin_label"].values)

# ─── Summary Table ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  Summary: BERT Fine-tuned Baseline")
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
