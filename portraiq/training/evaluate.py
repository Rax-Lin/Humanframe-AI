
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
        return {"mae": 0.0, "rmse": 0.0, "band_acc": 0.0}

    abs_errors = [abs(p - t) for p, t in zip(preds, targets)]
    sq_errors = [(p - t) ** 2 for p, t in zip(preds, targets)]

    def band(score: float) -> int:
        if score < 5.0:
            return 0
        if score < 7.5:
            return 1
        if score < 9.5:
            return 2
        return 3

    band_acc = sum(int(band(p) == band(t)) for p, t in zip(preds, targets)) / len(preds)

    return {
        "mae": sum(abs_errors) / len(abs_errors),
        "rmse": sqrt(sum(sq_errors) / len(sq_errors)),
        "band_acc": band_acc,
    }
