from __future__ import annotations

import torch
from torch import nn


class TabularTokenizer(nn.Module):
    """
    Tabular (batch, m) -> token embeddings (batch, m, d_model)

    Implementation (per PROJECT_PLAN.md):
      - value embedding: Linear(1 -> d_model) applied per feature scalar
      - feature-id embedding: Embedding(m -> d_model) added to each feature token
    """

    def __init__(self, num_features: int, d_model: int) -> None:
        super().__init__()
        self.num_features = int(num_features)
        self.d_model = int(d_model)
        self.value_proj = nn.Linear(1, d_model)
        self.feature_id = nn.Embedding(self.num_features, d_model)

        # Register feature indices so forward stays device-safe.
        self.register_buffer("feature_idx", torch.arange(self.num_features), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
          x: float tensor of shape (batch, m)
        Returns:
          tokens: float tensor of shape (batch, m, d_model)
        """
        if x.ndim != 2 or x.shape[1] != self.num_features:
            raise ValueError(f"Expected x shape (batch, {self.num_features}), got {tuple(x.shape)}")

        v = self.value_proj(x.unsqueeze(-1))  # (b, m, d)
        f = self.feature_id(self.feature_idx).unsqueeze(0)  # (1, m, d)
        return v + f

