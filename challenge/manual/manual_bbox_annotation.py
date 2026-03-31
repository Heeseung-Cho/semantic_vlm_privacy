from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def discover_images(image_dir: str | Path, patterns: Iterable[str] | None = None) -> list[Path]:
    root = Path(image_dir).resolve()
    patterns = tuple(patterns or ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp'))
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(root.glob(pattern))
    return sorted({path.resolve() for path in paths})


def load_annotation_store(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {
            'format': 'jupyter-bbox-widget-v1',
            'created_at': utc_now(),
            'updated_at': utc_now(),
            'classes': [],
            'image_root': None,
            'records': [],
        }
    return json.loads(path.read_text())


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_record_index(store: dict[str, Any]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for record in store.get('records', []):
        image_path = record.get('image_path')
        if image_path:
            index[image_path] = record
    return index


def ensure_record(store: dict[str, Any], image_path: Path, image_root: Path) -> dict[str, Any]:
    rel_path = str(image_path.resolve().relative_to(image_root.resolve()))
    index = build_record_index(store)
    record = index.get(rel_path)
    if record is not None:
        return record
    record = {
        'image_path': rel_path,
        'bboxes': [],
        'updated_at': utc_now(),
    }
    store.setdefault('records', []).append(record)
    return record


def normalize_widget_bboxes(bboxes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for bbox in bboxes:
        normalized.append({
            'x': int(round(float(bbox['x']))),
            'y': int(round(float(bbox['y']))),
            'width': int(round(float(bbox['width']))),
            'height': int(round(float(bbox['height']))),
            'label': str(bbox['label']),
        })
    return normalized


def save_record(store: dict[str, Any], image_path: Path, image_root: Path, classes: list[str], bboxes: list[dict[str, Any]]) -> dict[str, Any]:
    record = ensure_record(store, image_path=image_path, image_root=image_root)
    record['bboxes'] = normalize_widget_bboxes(bboxes)
    record['updated_at'] = utc_now()
    store['classes'] = list(classes)
    store['image_root'] = str(image_root.resolve())
    store['updated_at'] = utc_now()
    return record


def write_store(store: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(store, ensure_ascii=False, indent=2) + '\n')
