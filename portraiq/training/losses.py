
import torch.nn as nn


def build_loss(name: str = "smooth_l1") -> nn.Module:
    key = name.lower()
    # Default to SmoothL1 (Huber) for robustness to noisy score labels.
    # Keep "mse" as a backward-compatible alias to avoid breaking old configs.
    if key in {"mse", "smooth_l1", "huber"}:
        return nn.SmoothL1Loss(beta=1.0)
    raise ValueError("Unsupported loss. Use 'smooth_l1' (or alias 'mse').")
