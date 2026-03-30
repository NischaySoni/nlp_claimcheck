"""
CheckMate – Joint Model for Explainable Claim Check-Worthiness
================================================================
Combines CoNet (contextual/BERT) and LiNet (linguistic features)
to jointly predict:
  1. Binary check-worthiness (check-worthy vs non-check-worthy)
  2. Six rationality labels (multi-label)
"""

import torch
import torch.nn as nn
from co_net import CoNet
from li_net import LiNet


class CheckMate(nn.Module):
    """
    CheckMate model architecture:
        CoNet → BERT CLS + 6 attention-head outputs (contextual)
        LiNet → POS/Dep embedding aggregation (linguistic)
        Combined → classification heads for binary label + 6 rationality labels
    """

    def __init__(
        self,
        bert_model_name="bert-base-uncased",
        num_rationality=6,
        hidden_dim=768,
        linet_output_dim=256,
        dropout=0.25,
    ):
        super().__init__()

        self.num_rationality = num_rationality

        # Sub-networks
        self.co_net = CoNet(bert_model_name, num_rationality, hidden_dim)
        self.li_net = LiNet(output_dim=linet_output_dim)

        # Binary check-worthiness head
        # Input: CLS (768) + LiNet (256) = 1024
        combined_dim = hidden_dim + linet_output_dim
        self.cw_head = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2),  # binary classification
        )

        # Rationality label heads (one per label)
        # Each takes the corresponding attention head output (768) → binary
        self.rationality_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 1),
            )
            for _ in range(num_rationality)
        ])

    def forward(self, input_ids, attention_mask, pos_ids, dep_ids):
        """
        Args:
            input_ids:      (batch, seq_len)
            attention_mask: (batch, seq_len)
            pos_ids:        (batch, seq_len)
            dep_ids:        (batch, seq_len)
        Returns:
            cw_logits:          (batch, 2)  – check-worthiness logits
            rationality_logits: (batch, 6)  – rationality label logits (sigmoid applied later)
        """
        # CoNet: contextual features
        cls_output, head_outputs = self.co_net(input_ids, attention_mask)
        # cls_output: (batch, 768)
        # head_outputs: (batch, 6, 768)

        # LiNet: linguistic features
        ling_output = self.li_net(pos_ids, dep_ids)  # (batch, 256)

        # Binary check-worthiness
        combined = torch.cat([cls_output, ling_output], dim=-1)  # (batch, 1024)
        cw_logits = self.cw_head(combined)  # (batch, 2)

        # Rationality labels
        rat_logits = []
        for i, head in enumerate(self.rationality_heads):
            head_feat = head_outputs[:, i, :]  # (batch, 768)
            logit = head(head_feat)            # (batch, 1)
            rat_logits.append(logit)

        rationality_logits = torch.cat(rat_logits, dim=-1)  # (batch, 6)

        return cw_logits, rationality_logits
