"""
Baseline 2: Fine-tuned BERT for Binary Claim Check-Worthiness Detection
========================================================================
Fine-tunes bert-base-uncased on the 'claim' column for:
  1. Binary check-worthiness classification (primary head)
  2. Six rationality label predictions (multi-label head, auxiliary)

Uses HuggingFace Trainer API with a custom model that wraps BERT and
adds both heads on top of the CLS token.
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import (
    BertModel,
    BertTokenizerFast,
    PreTrainedModel,
    BertConfig,
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
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "bert_baseline_output")

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
MODEL_NAME    = "bert-base-uncased"
MAX_LEN       = 128
BATCH_SIZE    = 8
LEARNING_RATE = 2e-5
EPOCHS        = 5
PATIENCE      = 2
RAT_LOSS_W    = 0.5   # weight for auxiliary rationality loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ─── Load Data ────────────────────────────────────────────────────────
print("=" * 60)
print("Baseline 2: Fine-tuned BERT (binary + rationality heads)")
print("=" * 60)

train_df = pd.read_csv(TRAIN_PATH)
val_df   = pd.read_csv(VAL_PATH)
test_df  = pd.read_csv(TEST_PATH)

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")


def get_rat(df):
    """Rationality array; zero out non-check-worthy rows."""
    arr = df[RATIONALITY_COLS].fillna(0).values.astype(np.float32)
    arr[df["bin_label"].values == 0] = 0.0
    return arr


# ─── Tokenizer ────────────────────────────────────────────────────────
tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)


# ─── Dataset ──────────────────────────────────────────────────────────
class ClaimDataset(Dataset):
    """
    Packs binary label + 6 rationality labels into a single float
    tensor of shape (7,) under the key 'labels'.

    Layout:  labels[:, 0]  = binary label (cast to long inside model)
             labels[:, 1:] = 6 rationality labels (float)

    This avoids the HuggingFace Trainer bug where multiple label keys
    get bundled into a tuple and break compute_metrics.
    """
    def __init__(self, texts, bin_labels, rat_labels, tokenizer, max_len=128):
        self.texts      = texts.tolist()
        self.bin_labels = bin_labels      # 1-D int array (N,)
        self.rat_labels = rat_labels      # 2-D float array (N, 6)
        self.tokenizer  = tokenizer
        self.max_len    = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        bin_t = torch.tensor([self.bin_labels[idx]], dtype=torch.float)  # (1,)
        rat_t = torch.tensor(self.rat_labels[idx],   dtype=torch.float)  # (6,)
        packed = torch.cat([bin_t, rat_t], dim=0)                        # (7,)

        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         packed,   # (7,) — single key, Trainer-safe
        }


train_dataset = ClaimDataset(
    train_df["claim"].fillna(""), train_df["bin_label"].values,
    get_rat(train_df), tokenizer, MAX_LEN,
)
val_dataset = ClaimDataset(
    val_df["claim"].fillna(""), val_df["bin_label"].values,
    get_rat(val_df), tokenizer, MAX_LEN,
)
test_dataset = ClaimDataset(
    test_df["claim"].fillna(""), test_df["bin_label"].values,
    get_rat(test_df), tokenizer, MAX_LEN,
)


# ─── Custom BERT Model with Two Heads ─────────────────────────────────
class BertDualHead(PreTrainedModel):
    """
    BERT with:
      - Primary head   : binary cross-entropy for check-worthiness
      - Auxiliary head : multi-label BCE for 6 rationality labels

    Forward receives 'labels' of shape (batch, 7):
        col 0    → binary label  (cast to long)
        cols 1–6 → rationality   (float)

    Loss = CrossEntropy(binary) + RAT_LOSS_W * BCEWithLogits(rationality)

    Returns logits of shape (batch, 8):
        cols 0–1 → binary logits
        cols 2–7 → rationality logits
    """
    config_class = BertConfig

    def __init__(self, config, num_rat=NUM_RAT, rat_loss_w=RAT_LOSS_W):
        super().__init__(config)
        self.bert       = BertModel(config)
        self.dropout    = nn.Dropout(config.hidden_dropout_prob)
        self.rat_loss_w = rat_loss_w
        self.bin_head   = nn.Linear(config.hidden_size, 2)
        self.rat_head   = nn.Linear(config.hidden_size, num_rat)
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,        # (batch, 7) packed float tensor
        **kwargs,
    ):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls     = self.dropout(outputs.pooler_output)   # (batch, 768)

        bin_logits = self.bin_head(cls)                 # (batch, 2)
        rat_logits = self.rat_head(cls)                 # (batch, 6)

        loss = None
        if labels is not None:
            bin_labels = labels[:, 0].long()            # (batch,)
            rat_labels = labels[:, 1:]                  # (batch, 6)
            ce_loss    = nn.CrossEntropyLoss()(bin_logits, bin_labels)
            bce_loss   = nn.BCEWithLogitsLoss()(rat_logits, rat_labels)
            loss       = ce_loss + self.rat_loss_w * bce_loss

        # Concatenate: [:, 0:2] binary logits | [:, 2:8] rationality logits
        combined_logits = torch.cat([bin_logits, rat_logits], dim=-1)  # (batch, 8)

        return (loss, combined_logits) if loss is not None else combined_logits


# ─── Load Model ───────────────────────────────────────────────────────
print("\nLoading BERT model with dual heads ...")
config = BertConfig.from_pretrained(MODEL_NAME)
model  = BertDualHead.from_pretrained(MODEL_NAME, config=config)


# ─── Metrics (used by Trainer during validation) ──────────────────────
def compute_metrics(eval_pred):
    """
    eval_pred.predictions : (N, 8) — combined logits
    eval_pred.label_ids   : (N, 7) — packed labels
    """
    logits, packed_labels = eval_pred

    # col 0 of packed_labels is the binary label
    bin_labels = packed_labels[:, 0].astype(int)
    bin_preds  = np.argmax(logits[:, :2], axis=-1)

    acc      = accuracy_score(bin_labels, bin_preds)
    macro_f1 = f1_score(bin_labels, bin_preds, average="macro")
    return {"accuracy": acc, "macro_f1": macro_f1}


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
def evaluate_split(name, dataset, y_bin_true):
    preds_out     = trainer.predict(dataset)
    logits        = preds_out.predictions   # (N, 8)
    packed_labels = preds_out.label_ids     # (N, 7)

    # ── Binary check-worthiness ──
    y_bin_pred = np.argmax(logits[:, :2], axis=-1)
    acc        = accuracy_score(y_bin_true, y_bin_pred)
    macro_f1   = f1_score(y_bin_true, y_bin_pred, average="macro")
    cw_f1      = f1_score(y_bin_true, y_bin_pred, pos_label=1)
    ncw_f1     = f1_score(y_bin_true, y_bin_pred, pos_label=0)

    print(f"\n{'─' * 50}")
    print(f"  {name} Results — Binary Check-Worthiness")
    print(f"{'─' * 50}")
    print(f"  Accuracy       : {acc:.4f}")
    print(f"  Macro F1       : {macro_f1:.4f}")
    print(f"  CW F1 (cls=1)  : {cw_f1:.4f}")
    print(f"  NCW F1 (cls=0) : {ncw_f1:.4f}")
    print(f"\n  Classification Report:\n")
    print(
        classification_report(
            y_bin_true, y_bin_pred,
            target_names=["Non-Check-Worthy", "Check-Worthy"],
        )
    )

    # ── Rationality labels (check-worthy subset only) ──
    rat_logits   = logits[:, 2:]                                         # (N, 6)
    rat_preds    = (torch.sigmoid(torch.tensor(rat_logits)) > 0.5).numpy().astype(int)
    rat_true     = packed_labels[:, 1:].astype(int)                      # (N, 6)

    cw_mask      = y_bin_true == 1
    rat_preds_cw = rat_preds[cw_mask]
    rat_true_cw  = rat_true[cw_mask]

    print(f"  {name} Results — Rationality Labels "
          f"(check-worthy subset, n={int(cw_mask.sum())})")
    print(f"  {'Label':<35s}  {'Macro F1':>9}")
    print(f"  {'─' * 46}")

    rat_f1s = []
    for i, col in enumerate(RATIONALITY_COLS):
        f1 = f1_score(
            rat_true_cw[:, i], rat_preds_cw[:, i],
            average="macro", zero_division=0,
        )
        rat_f1s.append(f1)
        print(f"  {col:<35s}: {f1:.4f}")

    return acc, macro_f1, cw_f1, ncw_f1, rat_f1s


val_metrics  = evaluate_split("Validation", val_dataset,  val_df["bin_label"].values)
test_metrics = evaluate_split("Test",       test_dataset, test_df["bin_label"].values)

# ─── Summary Table ────────────────────────────────────────────────────
print("\n" + "=" * 95)
print("  Summary: BERT Fine-tuned Baseline (binary + rationality heads)")
print("=" * 95)
header = (
    f"  {'Split':<12} {'Acc':>6} {'m-F1':>6} {'cw-F1':>7} {'ncw-F1':>8}  "
    + "  ".join(f"R{i+1}" for i in range(NUM_RAT))
)
print(header)
print(f"  {'─' * 88}")
for split_name, m in [("Validation", val_metrics), ("Test", test_metrics)]:
    r = m[4]
    rat_str = "  ".join(f"{v:.4f}" for v in r)
    print(
        f"  {split_name:<12} {m[0]:>6.4f} {m[1]:>6.4f} {m[2]:>7.4f} {m[3]:>8.4f}  {rat_str}"
    )
print("=" * 95)
