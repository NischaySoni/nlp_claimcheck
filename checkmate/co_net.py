"""
CoNet – Contextual Network
===========================
BERT-based module with 6 attention heads (one per rationality label).
Extracts contextual representations from the input claim text.
"""

import torch
import torch.nn as nn
from transformers import BertModel


class CoNet(nn.Module):
    """
    Contextual Network based on BERT.

    Architecture:
        1. BERT encoder produces hidden states (768-dim)
        2. 6 separate linear attention heads (one per rationality label)
           each produce a weighted summary of the token representations
        3. CLS token output from BERT is also retained
    """

    def __init__(self, bert_model_name="bert-base-uncased", num_rationality=6, hidden_dim=768):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.num_rationality = num_rationality
        self.hidden_dim = hidden_dim

        # One attention head per rationality label
        # Each head: linear projection → softmax attention → weighted sum
        self.attn_heads = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(num_rationality)
        ])

        self.dropout = nn.Dropout(0.25)

    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids:      (batch, seq_len)
            attention_mask:  (batch, seq_len)
        Returns:
            cls_output:      (batch, hidden_dim)   – CLS token
            head_outputs:    (batch, num_rationality, hidden_dim) – per-head weighted sums
        """
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = bert_out.last_hidden_state   # (batch, seq_len, 768)
        cls_output = hidden_states[:, 0, :]          # (batch, 768)

        head_outputs = []
        for i, attn in enumerate(self.attn_heads):
            # Compute attention scores
            scores = attn(hidden_states).squeeze(-1)  # (batch, seq_len)
            # Mask padding tokens
            scores = scores.masked_fill(attention_mask == 0, float("-inf"))
            weights = torch.softmax(scores, dim=-1).unsqueeze(-1)  # (batch, seq_len, 1)
            # Weighted sum
            weighted = (hidden_states * weights).sum(dim=1)  # (batch, 768)
            head_outputs.append(weighted)

        head_outputs = torch.stack(head_outputs, dim=1)  # (batch, 6, 768)

        return self.dropout(cls_output), self.dropout(head_outputs)
