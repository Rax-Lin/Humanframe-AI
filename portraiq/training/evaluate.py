
from math import sqrt
from typing import Dict

import torch


def evaluate_model(model, dataloader, device) -> Dict[str, float]:
    model.eval()
    preds = []
    targets = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device, non_blocking=True)
            scores = batch["score"].to(device, non_blocking=True)
            outputs = model(images)
            preds.extend(outputs.detach().cpu().tolist())
            targets.extend(scores.detach().cpu().tolist())

    if not preds:
        return {"mae": 0.0, "rmse": 0.0}

    abs_errors = [abs(p - t) for p, t in zip(preds, targets)]
    sq_errors = [(p - t) ** 2 for p, t in zip(preds, targets)]

    return {
        "mae": sum(abs_errors) / len(abs_errors),
        "rmse": sqrt(sum(sq_errors) / len(sq_errors)),
    }
