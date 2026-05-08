
from pathlib import Path
from typing import Dict, List

from inference.predict import InferenceEngine
from inference.visualize import save_visualization


SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def run_batch_inference(
    input_dir: str,
    output_dir: str,
    config: Dict,
    checkpoint_path: str,
    detector_mode: str = "auto",
    cpu_optimized: bool = False,
    save_overlay: bool = True,
) -> List[Dict]:
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    engine = InferenceEngine(
        config=config,
        checkpoint_path=checkpoint_path,
        detector_mode=detector_mode,
        cpu_optimized=cpu_optimized,
    )

    results = []
    for image_path in sorted(in_dir.iterdir()):
        if image_path.suffix.lower() not in SUPPORTED_EXT:
            continue

        pred = engine.predict(str(image_path))
        if save_overlay:
            save_visualization(pred["overlay"], out_dir / image_path.name)
        pred.pop("overlay")
        results.append(pred)

    return results
