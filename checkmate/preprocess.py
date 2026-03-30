"""
CheckMate – Preprocessing & Dataset
====================================
Handles tokenization (BERT), linguistic feature extraction (SpaCy POS & Dep),
and creates PyTorch datasets for the CheckMate model.
"""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast
import spacy

# ─── Constants ────────────────────────────────────────────────────────
RATIONALITY_COLS = [
    "verifiable_factual_claim",
    "false_info",
    "general_public_interest",
    "harmful",
    "fact_checker_interest",
    "govt_interest",
]
MAX_LEN = 128
BERT_MODEL = "bert-base-uncased"

# Load SpaCy model (small, fast)
nlp = spacy.load("en_core_web_sm")
tokenizer = BertTokenizerFast.from_pretrained(BERT_MODEL)

# Fixed vocabulary sizes for POS and DEP tags
NUM_POS_TAGS = 100   # SpaCy fine-grained POS tag IDs (upper bound)
NUM_DEP_TAGS = 150   # SpaCy dependency label IDs (upper bound)


class CheckItDataset(Dataset):
    """
    PyTorch Dataset for the CheckIt corpus.
    Returns:
        - input_ids, attention_mask  (for BERT / CoNet)
        - pos_ids, dep_ids           (for LiNet)
        - bin_label                  (check-worthy: 0 or 1)
        - rationality_labels         (6 binary labels, float tensor)
    """

    def __init__(self, csv_path, max_len=MAX_LEN):
        df = pd.read_csv(csv_path)

        self.claims = df["claim"].fillna("").astype(str).tolist()
        self.bin_labels = df["bin_label"].values.astype(np.int64)

        # Rationality labels – fill NaN with 0
        rat_data = df[RATIONALITY_COLS].fillna(0).values.astype(np.float32)
        # For non-check-worthy samples, rationality labels should be 0
        for i in range(len(rat_data)):
            if self.bin_labels[i] == 0:
                rat_data[i] = 0.0
        self.rationality = rat_data

        self.max_len = max_len

        # Pre-tokenize with BERT
        self.encodings = tokenizer(
            self.claims,
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="np",
        )

        # Pre-extract linguistic features via SpaCy
        self.pos_ids_list = []
        self.dep_ids_list = []
        for claim in self.claims:
            doc = nlp(claim[:512])  # truncate for SpaCy
            pos_ids = [token.pos for token in doc][:max_len]
            dep_ids = [token.dep for token in doc][:max_len]
            # Pad to max_len
            pos_ids += [0] * (max_len - len(pos_ids))
            dep_ids += [0] * (max_len - len(dep_ids))
            self.pos_ids_list.append(pos_ids)
            self.dep_ids_list.append(dep_ids)

        self.pos_ids_arr = np.array(self.pos_ids_list, dtype=np.int64)
        self.dep_ids_arr = np.array(self.dep_ids_list, dtype=np.int64)

    def __len__(self):
        return len(self.claims)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.encodings["input_ids"][idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.encodings["attention_mask"][idx], dtype=torch.long),
            "pos_ids": torch.tensor(self.pos_ids_arr[idx], dtype=torch.long),
            "dep_ids": torch.tensor(self.dep_ids_arr[idx], dtype=torch.long),
            "bin_label": torch.tensor(self.bin_labels[idx], dtype=torch.long),
            "rationality_labels": torch.tensor(self.rationality[idx], dtype=torch.float),
        }


def load_datasets(data_dir):
    """Load train, val, and test datasets."""
    print("Loading and preprocessing datasets (this may take a minute) ...")
    train_ds = CheckItDataset(os.path.join(data_dir, "train_data.csv"))
    val_ds = CheckItDataset(os.path.join(data_dir, "val_data.csv"))
    test_ds = CheckItDataset(os.path.join(data_dir, "test_data.csv"))
    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    return train_ds, val_ds, test_ds
