from pathlib import Path
from typing import Dict, Optional

import torch

from inference.scoring_rules import compute_rule_based_score
from inference.visualize import draw_scoring_overlay
from models.modeling import build_composition_model
from utils.image_utils import build_model_transform, load_pil_rgb, pil_to_rgb_array
from utils.pose_utils import PersonDetector


class InferenceEngine:
    def __init__(
        self,
        config: Dict,
        checkpoint_path: str,
        detector_mode: str = "auto",
        cpu_optimized: bool = False,
    ) -> None:
        self.config = config
        self.device = self._resolve_device(config)

        self.model, _ = build_composition_model(config)
        self.model = self.model.to(self.device)
        self._load_checkpoint(checkpoint_path)

        if cpu_optimized and self.device.type == "cpu":
            self.model = self._optimize_for_cpu(self.model)

        backbone_name = config["model"]["backbone"]
        self.transform = build_model_transform(
            image_size=int(config["data"]["image_size"]),
            backbone_name=backbone_name,
            train=False,
        )

        use_yolo = detector_mode in {"auto", "yolo"}
        self.detector = PersonDetector(use_yolo=use_yolo)
        self.alpha = float(config["scoring"].get("alpha", 0.3))

    def _resolve_device(self, config: Dict):
        wants_cuda = config.get("gpu", {}).get("device", "cuda") == "cuda"
        if wants_cuda and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _load_checkpoint(self, checkpoint_path: str):
        payload = torch.load(checkpoint_path, map_location=self.device)
        state_dict = payload.get("model", payload)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def _optimize_for_cpu(self, model):
        torch.set_num_threads(max(1, int(self.config.get("inference", {}).get("cpu_threads", 2))))
        quantized = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        return quantized

    def predict(self, image_path: str) -> Dict:
        image_pil = load_pil_rgb(Path(image_path))
        image_rgb = pil_to_rgb_array(image_pil)

        detect = self.detector.detect_largest_person(image_rgb)

        image_tensor = self.transform(image_pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            ai_score = float(self.model(image_tensor).item())

        rule = compute_rule_based_score(detect.bbox, image_size=(image_rgb.shape[0], image_rgb.shape[1]))
        rule_score = float(rule["score"])
        final_score = self.alpha * rule_score + (1.0 - self.alpha) * ai_score

        overlay = draw_scoring_overlay(
            image_pil,
            bbox=detect.bbox,
            final_score=final_score,
            ai_score=ai_score,
            rule_score=rule_score,
        )

        return {
            "image_path": image_path,
            "bbox": detect.bbox,
            "detector": detect.source,
            "detector_confidence": detect.confidence,
            "ai_score": round(ai_score, 4),
            "rule_score": round(rule_score, 4),
            "final_score": round(final_score, 4),
            "sub_scores": rule["sub_scores"],
            "overlay": overlay,
        }


def predict_single_image(
    image_path: str,
    config: Dict,
    checkpoint_path: str,
    detector: Optional[PersonDetector] = None,
):
    engine = InferenceEngine(config=config, checkpoint_path=checkpoint_path)
    if detector is not None:
        engine.detector = detector
    return engine.predict(image_path)
