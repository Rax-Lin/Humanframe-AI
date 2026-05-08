from pathlib import Path
from typing import Dict

from PIL import Image, ImageDraw


def _draw_thirds(draw: ImageDraw.ImageDraw, w: int, h: int):
    color = (180, 180, 180)
    draw.line([(w // 3, 0), (w // 3, h)], fill=color, width=1)
    draw.line([(2 * w // 3, 0), (2 * w // 3, h)], fill=color, width=1)
    draw.line([(0, h // 3), (w, h // 3)], fill=color, width=1)
    draw.line([(0, 2 * h // 3), (w, 2 * h // 3)], fill=color, width=1)


def draw_scoring_overlay(image_rgb: Image.Image, bbox: Dict[str, float], final_score: float, ai_score: float, rule_score: float):
    output = image_rgb.copy()
    draw = ImageDraw.Draw(output)

    w, h = output.size
    _draw_thirds(draw, w, h)

    x1, y1, x2, y2 = int(bbox["x1"]), int(bbox["y1"]), int(bbox["x2"]), int(bbox["y2"])
    draw.rectangle([(x1, y1), (x2, y2)], outline=(40, 220, 40), width=2)

    text_lines = [
        "Final: {:.2f}/10".format(final_score),
        "AI: {:.2f}".format(ai_score),
        "Rules: {:.2f}".format(rule_score),
    ]
    y = 12
    for line in text_lines:
        draw.text((12, y), line, fill=(255, 255, 255))
        y += 18

    return output


def save_visualization(image_rgb: Image.Image, out_path: Path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image_rgb.save(str(out_path))
