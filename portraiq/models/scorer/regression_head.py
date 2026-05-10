
import torch
import torch.nn as nn


class RegressionHead(nn.Module):
    """Predicts composition score in [0, 10]."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.3) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        raw = self.layers(features).squeeze(-1)
        return torch.clamp(raw, 0.0, 10.0)
