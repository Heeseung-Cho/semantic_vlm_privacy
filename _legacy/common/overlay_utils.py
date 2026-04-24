from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from PIL import Image, ImageDraw, ImageOps


def load_display_image(image_path: str | Path) -> Image.Image:
    return ImageOps.exif_transpose(Image.open(image_path)).convert('RGB')


def xywh_to_xyxy(bbox: Sequence[float]) -> list[float]:
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


def draw_gt_annotation(
    draw: ImageDraw.ImageDraw,
    ann: dict[str, Any],
    label: str | None = None,
    bbox_color: str = 'lime',
    polygon_color: str = 'yellow',
) -> None:
    x1, y1, x2, y2 = xywh_to_xyxy(ann['bbox'])
    draw.rectangle([x1, y1, x2, y2], outline=bbox_color, width=4)
    text = f'GT {label}'.strip() if label else 'GT'
    draw.text((x1, max(0, y1 - 18)), text, fill=bbox_color)
    for polygon in ann.get('segmentation', []):
        pts = [(polygon[i], polygon[i + 1]) for i in range(0, len(polygon), 2)]
        if len(pts) >= 2:
            draw.line(pts + [pts[0]], fill=polygon_color, width=3)


def draw_xyxy_box(
    draw: ImageDraw.ImageDraw,
    xyxy: Sequence[float],
    label: str,
    color: str = 'red',
    width: int = 4,
) -> None:
    x1, y1, x2, y2 = xyxy
    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
    draw.text((x1, max(0, y1 - 18)), label, fill=color)


def draw_xywh_box(
    draw: ImageDraw.ImageDraw,
    bbox: Sequence[float],
    label: str,
    color: str = 'red',
    width: int = 4,
) -> None:
    x, y, w, h = bbox
    draw.rectangle([x, y, x + w, y + h], outline=color, width=width)
    draw.text((x, max(0, y - 18)), label, fill=color)


def draw_segmentation_polygons(
    draw: ImageDraw.ImageDraw,
    polygons: Sequence[Sequence[float]],
    color: str = 'cyan',
    width: int = 2,
) -> None:
    for polygon in polygons:
        pts = [(polygon[i], polygon[i + 1]) for i in range(0, len(polygon), 2)]
        if len(pts) >= 2:
            draw.line(pts + [pts[0]], fill=color, width=width)
