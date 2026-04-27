from __future__ import annotations

from typing import Literal, Optional

import torch
from torch import nn

from src.models.tabular_tokenizer import TabularTokenizer


Pooling = Literal["mean", "cls"]


class TransformerFraudClassifier(nn.Module):
    def __init__(
        self,
        num_features: int,
        d_model: int = 64,
        nhead: int = 6,
        num_layers: int = 2,
        dim_feedforward: int = 1024,
        dropout: float = 0.2,
        pooling: Pooling = "mean",
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.num_features = int(num_features)
        self.d_model = int(d_model)
        self.pooling: Pooling = pooling

        self.tokenizer = TabularTokenizer(num_features=self.num_features, d_model=self.d_model)

        if self.pooling == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        else:
            self.cls_token = None

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(nhead),
            dim_feedforward=int(dim_feedforward),
            dropout=float(dropout),
            activation="relu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(num_layers))
        self.dropout = nn.Dropout(float(dropout))
        self.head = nn.Linear(self.d_model, int(num_classes))

        self._init_parameters()

    def _init_parameters(self) -> None:
        # Keep initialization stable; PyTorch already initializes most layers well.
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
          x: (batch, m) float tensor
        Returns:
          logits: (batch, num_classes)
        """
        tokens = self.tokenizer(x)  # (b, m, d)
        if self.pooling == "cls":
            assert self.cls_token is not None
            b = tokens.shape[0]
            cls = self.cls_token.expand(b, -1, -1)  # (b, 1, d)
            tokens = torch.cat([cls, tokens], dim=1)  # (b, 1+m, d)

        h = self.encoder(tokens)  # (b, seq, d)

        if self.pooling == "mean":
            pooled = h.mean(dim=1)
        elif self.pooling == "cls":
            pooled = h[:, 0, :]
        else:
            raise ValueError(f"Unknown pooling '{self.pooling}'")

        pooled = self.dropout(pooled)
        return self.head(pooled)

