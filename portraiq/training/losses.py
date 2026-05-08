
import torch.nn as nn


def build_loss(name: str = "mse") -> nn.Module:
    key = name.lower()
    if key == "mse":
        return nn.MSELoss()
    if key in {"smooth_l1", "huber"}:
        return nn.SmoothL1Loss(beta=1.0)
    raise ValueError("Unsupported loss. Use 'mse' or 'smooth_l1'.")
