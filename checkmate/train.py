"""
CheckMate – Training Script
=============================
Trains the CheckMate model for joint check-worthiness + rationality
label prediction. Reports Accuracy, macro-F1, class-wise F1 for
check-worthiness and F1 for each rationality label.
"""

import os
import sys
import copy
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Local modules
from preprocess import load_datasets, RATIONALITY_COLS
from checkmate import CheckMate

# ─── Config ───────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters (from the paper)
EPOCHS = 20
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-3
BERT_LR = 2e-5       # lower LR for BERT parameters
PATIENCE = 5
DROPOUT = 0.25

# Loss weights
CW_LOSS_WEIGHT = 1.0
RAT_LOSS_WEIGHT = 0.5  # weight for rationality label loss

print(f"Device: {DEVICE}")
print("=" * 60)
print("  CheckMate – Training")
print("=" * 60)

# ─── Data ─────────────────────────────────────────────────────────────
train_ds, val_ds, test_ds = load_datasets(DATA_DIR)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ─── Model ────────────────────────────────────────────────────────────
model = CheckMate(dropout=DROPOUT).to(DEVICE)

# Separate parameter groups for BERT vs rest (different LRs)
bert_params = list(model.co_net.bert.parameters())
other_params = [p for n, p in model.named_parameters() if "co_net.bert" not in n]

optimizer = optim.Adam([
    {"params": bert_params, "lr": BERT_LR},
    {"params": other_params, "lr": LEARNING_RATE},
], weight_decay=WEIGHT_DECAY)

# Learning rate scheduler
scheduler = optim.lr_scheduler.LinearLR(
    optimizer, start_factor=1.0, end_factor=0.1, total_iters=EPOCHS
)

# Loss functions
cw_criterion = nn.CrossEntropyLoss()
rat_criterion = nn.BCEWithLogitsLoss()


# ─── Evaluate ─────────────────────────────────────────────────────────
def evaluate(model, loader, verbose=False):
    """Evaluate model on a dataloader. Returns dict of metrics."""
    model.eval()
    all_cw_preds, all_cw_labels = [], []
    all_rat_preds, all_rat_labels = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            pos_ids = batch["pos_ids"].to(DEVICE)
            dep_ids = batch["dep_ids"].to(DEVICE)
            bin_labels = batch["bin_label"].to(DEVICE)
            rat_labels = batch["rationality_labels"].to(DEVICE)

            cw_logits, rat_logits = model(input_ids, attention_mask, pos_ids, dep_ids)

            cw_preds = torch.argmax(cw_logits, dim=1).cpu().numpy()
            rat_preds = (torch.sigmoid(rat_logits) > 0.5).float().cpu().numpy()

            all_cw_preds.extend(cw_preds)
            all_cw_labels.extend(bin_labels.cpu().numpy())
            all_rat_preds.extend(rat_preds)
            all_rat_labels.extend(rat_labels.cpu().numpy())

    all_cw_preds = np.array(all_cw_preds)
    all_cw_labels = np.array(all_cw_labels)
    all_rat_preds = np.array(all_rat_preds)
    all_rat_labels = np.array(all_rat_labels)

    # Check-worthiness metrics
    acc = accuracy_score(all_cw_labels, all_cw_preds)
    macro_f1 = f1_score(all_cw_labels, all_cw_preds, average="macro")
    cw_f1 = f1_score(all_cw_labels, all_cw_preds, pos_label=1)
    ncw_f1 = f1_score(all_cw_labels, all_cw_preds, pos_label=0)

    # Rationality label metrics (macro F1 per label)
    rat_f1s = []
    for i in range(6):
        f1 = f1_score(all_rat_labels[:, i], all_rat_preds[:, i], average="macro", zero_division=0)
        rat_f1s.append(f1)

    metrics = {
        "acc": acc,
        "macro_f1": macro_f1,
        "cw_f1": cw_f1,
        "ncw_f1": ncw_f1,
        "rat_f1s": rat_f1s,
    }

    if verbose:
        print(f"  Accuracy       : {acc:.4f}")
        print(f"  Macro F1       : {macro_f1:.4f}")
        print(f"  CW F1 (cls=1)  : {cw_f1:.4f}")
        print(f"  NCW F1 (cls=0) : {ncw_f1:.4f}")
        print(f"  Rationality F1s:")
        for i, name in enumerate(RATIONALITY_COLS):
            print(f"    {name:<30s}: {rat_f1s[i]:.4f}")
        print(f"\n  Classification Report (Check-Worthiness):\n")
        print(classification_report(
            all_cw_labels, all_cw_preds,
            target_names=["Non-Check-Worthy", "Check-Worthy"],
        ))

    return metrics


# ─── Training Loop ────────────────────────────────────────────────────
best_val_f1 = 0.0
best_model_state = None
patience_counter = 0

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in train_loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        pos_ids = batch["pos_ids"].to(DEVICE)
        dep_ids = batch["dep_ids"].to(DEVICE)
        bin_labels = batch["bin_label"].to(DEVICE)
        rat_labels = batch["rationality_labels"].to(DEVICE)

        # Forward
        cw_logits, rat_logits = model(input_ids, attention_mask, pos_ids, dep_ids)

        # Losses
        cw_loss = cw_criterion(cw_logits, bin_labels)
        rat_loss = rat_criterion(rat_logits, rat_labels)
        loss = CW_LOSS_WEIGHT * cw_loss + RAT_LOSS_WEIGHT * rat_loss

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    scheduler.step()
    avg_loss = total_loss / num_batches

    # Validate
    val_metrics = evaluate(model, val_loader)
    val_f1 = val_metrics["macro_f1"]

    print(
        f"Epoch {epoch+1:>2}/{EPOCHS} | "
        f"Loss: {avg_loss:.4f} | "
        f"Val Acc: {val_metrics['acc']:.4f} | "
        f"Val m-F1: {val_f1:.4f}"
    )

    # Early stopping
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_model_state = copy.deepcopy(model.state_dict())
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch+1} (patience={PATIENCE})")
            break

# ─── Load best model & evaluate on test ───────────────────────────────
if best_model_state is not None:
    model.load_state_dict(best_model_state)

print("\n" + "=" * 60)
print("  Validation Results (Best Model)")
print("=" * 60)
val_metrics = evaluate(model, val_loader, verbose=True)

print("\n" + "=" * 60)
print("  Test Results (Best Model)")
print("=" * 60)
test_metrics = evaluate(model, test_loader, verbose=True)

# ─── Summary Table ────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  Summary: CheckMate")
print("=" * 70)
print(f"  {'Split':<12} {'Acc':>8} {'m-F1':>8} {'cw-F1':>8} {'ncw-F1':>8}  R1     R2     R3     R4     R5     R6")
print(f"  {'─'*90}")
for name, m in [("Validation", val_metrics), ("Test", test_metrics)]:
    r = m["rat_f1s"]
    print(
        f"  {name:<12} {m['acc']:>8.4f} {m['macro_f1']:>8.4f} "
        f"{m['cw_f1']:>8.4f} {m['ncw_f1']:>8.4f}  "
        f"{r[0]:.4f} {r[1]:.4f} {r[2]:.4f} {r[3]:.4f} {r[4]:.4f} {r[5]:.4f}"
    )
print("=" * 70)