
import os
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.modeling import build_composition_model
from training.augment import build_eval_transform, build_train_transform
from training.dataset import CompositionDataset, load_annotation_records, split_records
from training.evaluate import evaluate_model
from training.losses import build_loss
from utils.gpu_config import create_grad_scaler, maybe_autocast, prepare_model
from utils.logger import create_logger


def _build_optimizer(model, config: Dict):
    base_lr = float(config["training"]["learning_rate"])
    wd = float(config["training"].get("weight_decay", 1e-5))
    lr_scale = float(config["model"].get("backbone_lr_scale", 0.1))

    params = [
        {"params": model.backbone.parameters(), "lr": base_lr * lr_scale},
        {"params": model.scorer_head.parameters(), "lr": base_lr},
    ]
    return torch.optim.AdamW(params, lr=base_lr, weight_decay=wd)


def _build_dataloaders(config: Dict):
    data_dir = Path("data")
    image_root = data_dir / "processed"
    anno_root = data_dir / "annotations"

    records = load_annotation_records(anno_root)
    train_r, val_r, test_r = split_records(
        records,
        train_ratio=float(config["data"]["train_split"]),
        val_ratio=float(config["data"]["val_split"]),
        test_ratio=float(config["data"]["test_split"]),
        seed=int(config["training"].get("seed", 42)),
    )

    image_size = int(config["data"]["image_size"])
    backbone_name = config["model"]["backbone"]
    train_ds = CompositionDataset(
        train_r,
        image_root=image_root,
        transform=build_train_transform(image_size=image_size, backbone_name=backbone_name),
    )
    val_ds = CompositionDataset(
        val_r,
        image_root=image_root,
        transform=build_eval_transform(image_size=image_size, backbone_name=backbone_name),
    )
    test_ds = CompositionDataset(
        test_r,
        image_root=image_root,
        transform=build_eval_transform(image_size=image_size, backbone_name=backbone_name),
    )

    if len(train_ds) == 0:
        raise RuntimeError(
            "No training samples found. Add annotation JSON files under data/annotations "
            "and ensure records are assigned to split='train' (or valid split ratios are set)."
        )

    batch_size = int(config["training"]["batch_size"])
    num_workers = int(config["data"].get("num_workers", 4))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader, test_loader


def _save_checkpoint(checkpoint_path: Path, model, optimizer, epoch: int, best_metric: float):
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_val_mae": best_metric,
        },
        checkpoint_path,
    )


def run_training(config: Dict, resume: Optional[str] = None) -> Dict[str, float]:
    logger = create_logger("train")
    device, model = _prepare_model_and_device(config)
    train_loader, val_loader, test_loader = _build_dataloaders(config)

    criterion = build_loss(config["training"].get("loss", "mse"))
    optimizer = _build_optimizer(model, config)
    scaler = create_grad_scaler(config, device)

    start_epoch = 0
    best_val_mae = float("inf")

    if resume:
        ckpt = torch.load(resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val_mae = float(ckpt.get("best_val_mae", best_val_mae))
        logger.info("Resumed from %s at epoch %d", resume, start_epoch)

    epochs = int(config["training"]["epochs"])
    checkpoint_dir = Path("models/checkpoints")

    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0

        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for batch in progress:
            images = batch["image"].to(device, non_blocking=True)
            scores = batch["score"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with maybe_autocast(config, device):
                preds = model(images)
                loss = criterion(preds, scores)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += float(loss.item())
            progress.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running_loss / max(1, len(train_loader))
        val_metrics = evaluate_model(model, val_loader, device)
        logger.info(
            "Epoch %d | train_loss=%.4f val_mae=%.4f val_rmse=%.4f",
            epoch + 1,
            train_loss,
            val_metrics["mae"],
            val_metrics["rmse"],
        )

        _save_checkpoint(checkpoint_dir / "last.pth", model, optimizer, epoch, best_val_mae)

        if val_metrics["mae"] < best_val_mae:
            best_val_mae = val_metrics["mae"]
            _save_checkpoint(checkpoint_dir / "best.pth", model, optimizer, epoch, best_val_mae)
            logger.info("New best checkpoint saved (val_mae=%.4f)", best_val_mae)

    test_metrics = evaluate_model(model, test_loader, device)
    logger.info("Test metrics: %s", test_metrics)
    return test_metrics


def run_evaluation(config: Dict, checkpoint: str) -> Dict[str, float]:
    logger = create_logger("eval")
    device, model = _prepare_model_and_device(config)

    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])

    _, _, test_loader = _build_dataloaders(config)
    metrics = evaluate_model(model, test_loader, device)
    logger.info("Evaluation metrics: %s", metrics)
    return metrics


def _prepare_model_and_device(config: Dict):
    device_cfg = config.get("gpu", {})
    use_cuda = device_cfg.get("device", "cuda") == "cuda" and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda and device_cfg.get("visible_devices"):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_cfg["visible_devices"])

    model, _ = build_composition_model(config)
    model = prepare_model(model, config, device)
    return device, model
