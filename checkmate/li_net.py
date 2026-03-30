"""
LiNet – Linguistic Network
============================
Extracts linguistic features from POS tags and dependency parse labels.
Embeds them and produces a fixed-dim linguistic representation.
"""

import torch
import torch.nn as nn


class LiNet(nn.Module):
    """
    Linguistic Network.

    Embeds POS tag IDs and Dependency label IDs into dense vectors,
    then combines them via concatenation + linear projection.
    """

    def __init__(self, num_pos=100, num_dep=150, embed_dim=64, output_dim=256):
        super().__init__()
        self.pos_embed = nn.Embedding(num_pos, embed_dim, padding_idx=0)
        self.dep_embed = nn.Embedding(num_dep, embed_dim, padding_idx=0)

        # Project combined embeddings to output_dim
        self.fc = nn.Sequential(
            nn.Linear(embed_dim * 2, output_dim),
            nn.ReLU(),
            nn.Dropout(0.25),
        )

    def forward(self, pos_ids, dep_ids):
        """
        Args:
            pos_ids: (batch, seq_len) – SpaCy POS tag integer IDs
            dep_ids: (batch, seq_len) – SpaCy dependency label integer IDs
        Returns:
            (batch, output_dim) – aggregated linguistic features
        """
        pos_emb = self.pos_embed(pos_ids)   # (batch, seq_len, embed_dim)
        dep_emb = self.dep_embed(dep_ids)   # (batch, seq_len, embed_dim)

        # Concatenate POS and Dep embeddings
        combined = torch.cat([pos_emb, dep_emb], dim=-1)  # (batch, seq_len, 2*embed_dim)

        # Average pool across sequence
        combined = combined.mean(dim=1)  # (batch, 2*embed_dim)

        return self.fc(combined)  # (batch, output_dim)
