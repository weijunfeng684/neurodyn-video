"""Stacked GRU + linear head; clip embedding = hidden at final time index."""

import torch
import torch.nn as nn


class GRUClassifier(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_layers,
        num_classes,
        dropout=0.0,
    ):
        super().__init__()

        # PyTorch GRU dropout only applies between stacked layers, not on the last layer output.
        gru_dropout = dropout if num_layers > 1 else 0.0

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=gru_dropout,
        )

        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: (B, T, D). Training code right-pads; no pack_padded_sequence here.
        hidden_seq, _ = self.gru(x)
        last_hidden = hidden_seq[:, -1, :]
        logits = self.classifier(last_hidden)
        return logits, hidden_seq
