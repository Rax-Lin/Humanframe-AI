
import json
import random
from pathlib import Path
from typing import Dict, List, Sequence

from PIL import Image
from torch.utils.data import Dataset


class CompositionDataset(Dataset):
    def __init__(self, records: Sequence[Dict], image_root: Path, transform=None) -> None:
        self.records = list(records)
        self.image_root = Path(image_root)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        image_path = self.image_root / record["filename"]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return {
            "image": image,
            "score": float(record["score"]),
            "filename": record["filename"],
            "image_id": record["image_id"],
        }


def _parse_json_annotation(payload: Dict) -> Dict:
    if "filename" not in payload or "score" not in payload:
        raise ValueError("Each annotation must include 'filename' and 'score'.")

    score = float(payload["score"])
    if score < 0.0 or score > 10.0:
        raise ValueError("Annotation score must be in [0.0, 10.0].")

    return {
        "filename": payload["filename"],
        "score": score,
        "split": payload.get("split", "unspecified"),
        "image_id": payload.get("image_id", Path(payload["filename"]).stem),
    }


def load_annotation_records(annotation_dir: Path) -> List[Dict]:
    annotation_dir = Path(annotation_dir)
    records: List[Dict] = []
    for path in sorted(annotation_dir.glob("*.json")):
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        if isinstance(payload, list):
            records.extend(_parse_json_annotation(item) for item in payload)
        else:
            records.append(_parse_json_annotation(payload))

    return records


def split_records(
    records: Sequence[Dict],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int = 42,
):
    if not records:
        return [], [], []

    explicit_splits = {r["split"] for r in records if r["split"] in {"train", "val", "test"}}
    if explicit_splits:
        train = [r for r in records if r["split"] == "train"]
        val = [r for r in records if r["split"] == "val"]
        test = [r for r in records if r["split"] == "test"]
        return train, val, test

    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("train/val/test split ratios must sum to 1.0")

    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train = shuffled[:n_train]
    val = shuffled[n_train : n_train + n_val]
    test = shuffled[n_train + n_val :]
    return train, val, test
