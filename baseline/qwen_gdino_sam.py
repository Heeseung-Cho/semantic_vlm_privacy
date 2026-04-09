from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import mmcv
import numpy as np
import torch
from torchvision import ops

from common.vlm import DEFAULT_INSTRUCTION, SwiftVLMCaller
from common.model_loaders import load_groundingdino_model, load_sam_model
from common.text_utils import parse_response, preprocess_caption


SUPPORT_QUERY_PROMPT = """
You are given support image(s) followed by one query image.
Use the support image(s) as reference examples of the target category.
Identify the category shared by the support image(s), then output the matching category names in the query image.
Return the final category list only inside <output>...</output>.
If nothing in the query image clearly matches the support references, return:
<output>No objects matching the given categories could be identified</output>
""".strip()

NEGATIVE_CUE_TEXTS = {
    'no output found',
    'no objects matching the given categories could be identified',
}

DESCRIPTIVE_SPLIT_PATTERNS = (
    r'\bsuggesting\b',
    r'\bshowing\b',
    r'\bwith\b',
    r'\bsitting\b',
    r'\blying\b',
    r'\bplaced\b',
    r'\bon top of\b',
    r'\bin the\b',
    r'\bon the\b',
    r'\bpartially\b',
    r'\bnext to\b',
)


@dataclass
class ControllerOutput:
    raw_text: str
    cue_text: str
    image_paths: list[str]
    instruction: str


@dataclass
class DetectionCandidate:
    score: float
    label_text: str
    category_id: int
    xyxy: list[float]


class QwenController:
    def __init__(
        self,
        model_path: str,
        max_new_tokens: int = 256,
        decoding_mode: str = 'deterministic',
        seed: int | None = None,
        max_pixels: int = 448,
        instruction: str | None = None,
        support_query_instruction: str | None = None,
        allowed_categories: Sequence[str] | None = None,
    ) -> None:
        self.client = SwiftVLMCaller(
            model_path=model_path,
            max_new_tokens=max_new_tokens,
            decoding_mode=decoding_mode,
            seed=seed,
            max_pixels=max_pixels,
        )
        self.query_instruction = instruction or DEFAULT_INSTRUCTION
        self.support_query_instruction = support_query_instruction or SUPPORT_QUERY_PROMPT
        self.allowed_categories = list(allowed_categories or [])

    def get_default_query_instruction(self) -> str:
        return self.query_instruction

    def get_default_support_query_instruction(self) -> str:
        return self.support_query_instruction

    def infer_query_only(self, query_image_path: str, instruction: str | None = None) -> ControllerOutput:
        resolved_instruction = instruction or self.query_instruction
        raw_text = self.client.generate(query_image_path, instruction=resolved_instruction)
        cue_text = sanitize_detector_cue(parse_response(raw_text), self.allowed_categories)
        return ControllerOutput(
            raw_text=raw_text,
            cue_text=cue_text,
            image_paths=[query_image_path],
            instruction=resolved_instruction,
        )

    def infer_support_query(
        self,
        support_image_paths: Sequence[str],
        query_image_path: str,
        instruction: str | None = None,
    ) -> ControllerOutput:
        resolved_instruction = instruction or self.support_query_instruction
        image_paths = [*map(str, support_image_paths), str(query_image_path)]
        raw_text = self.client.generate_images(image_paths, instruction=resolved_instruction)
        cue_text = sanitize_detector_cue(parse_response(raw_text), self.allowed_categories)
        return ControllerOutput(
            raw_text=raw_text,
            cue_text=cue_text,
            image_paths=image_paths,
            instruction=resolved_instruction,
        )


def sanitize_detector_cue(cue_text: str, allowed_categories: Sequence[str] | None = None) -> str:
    cleaned_cue = (cue_text or "").strip()
    lowered_cue = cleaned_cue.lower()
    if not cleaned_cue or lowered_cue in NEGATIVE_CUE_TEXTS:
        return cue_text

    allowed_categories = list(allowed_categories or [])
    normalized_allowed = {normalize_phrase(category): category for category in allowed_categories}

    normalized_items = []
    for item in cleaned_cue.replace("\n", ",").split(","):
        normalized_item = normalize_phrase(item)
        if normalized_item:
            normalized_items.append(normalized_item)

    if not normalized_items:
        return "No output found"

    if not allowed_categories:
        return ", ".join(dedupe_preserve_order(normalized_items))

    matched_categories = []
    for normalized_item in normalized_items:
        matched_category = match_allowed_category(normalized_item, normalized_allowed)
        if matched_category:
            matched_categories.append(matched_category)

    if matched_categories:
        return ", ".join(dedupe_preserve_order(matched_categories))

    return "No output found"


def normalize_phrase(text: str) -> str:
    phrase = (text or "").strip().strip("[](){}")
    phrase = re.sub(r"^\d+\.\s*", "", phrase)
    phrase = re.sub(r"\s+", " ", phrase)
    lowered = phrase.lower()
    for pattern in DESCRIPTIVE_SPLIT_PATTERNS:
        match = re.search(pattern, lowered)
        if match:
            phrase = phrase[:match.start()].strip(" ,.;:-")
            break
    phrase = re.sub(r"[.;:]+$", "", phrase.strip())
    return phrase


def match_allowed_category(normalized_item: str, normalized_allowed: dict[str, str]) -> str | None:
    if normalized_item in normalized_allowed:
        return normalized_allowed[normalized_item]

    for normalized_category, original_category in normalized_allowed.items():
        if normalized_category in normalized_item or normalized_item in normalized_category:
            return original_category

    matches = difflib.get_close_matches(normalized_item, list(normalized_allowed.keys()), n=1, cutoff=0.72)
    if matches:
        return normalized_allowed[matches[0]]
    return None


def dedupe_preserve_order(items: Sequence[str]) -> list[str]:
    deduped = []
    seen = set()
    for item in items:
        lowered = item.lower()
        if lowered in seen:
            continue
        deduped.append(item)
        seen.add(lowered)
    return deduped


class GroundingDinoLocalizer:
    def __init__(self, config_path: str, checkpoint_path: str, device: str = 'cuda') -> None:
        self.model = load_groundingdino_model(config_path=config_path, checkpoint_path=checkpoint_path, device=device)

    def detect(
        self,
        image_path: str,
        cue_text: str,
        categories_dict: dict[str, int],
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
        nms_iou: float = 0.8,
        max_dets: int = 100,
    ) -> list[DetectionCandidate]:
        if not cue_text or cue_text == 'No output found':
            return []
        image = mmcv.imread(image_path, channel_order='rgb')
        prompt = preprocess_caption(cue_text)
        label_texts = [cls.strip() for cls in cue_text.split(',') if cls.strip()]
        if not label_texts:
            return []

        result = self.model(inputs=image, texts=[prompt])
        if isinstance(result, list):
            result = result[0]
        if not isinstance(result, dict) or 'predictions' not in result:
            return []
        predictions = result['predictions']
        if not predictions or not isinstance(predictions[0], dict):
            return []
        first_pred = predictions[0]
        boxes = first_pred.get('bboxes', [])
        scores = first_pred.get('scores', [])
        if not isinstance(boxes, torch.Tensor):
            boxes = torch.tensor(boxes)
        if not isinstance(scores, torch.Tensor):
            scores = torch.tensor(scores)
        if len(boxes) == 0 or len(scores) == 0:
            return []

        top_indices = torch.argsort(scores, descending=True)[:max_dets]
        keep_indices_relative = ops.nms(boxes[top_indices], scores[top_indices], iou_threshold=nms_iou)
        final_indices = top_indices[keep_indices_relative]

        detections: list[DetectionCandidate] = []
        for idx in final_indices.tolist():
            score = float(scores[idx].item())
            if score < box_threshold or score < text_threshold:
                continue
            label_text = label_texts[idx] if idx < len(label_texts) else label_texts[0]
            if label_text not in categories_dict:
                matched = next((cls for cls in label_texts if label_text in cls), None)
                if matched is None or matched not in categories_dict:
                    continue
                label_text = matched
            detections.append(
                DetectionCandidate(
                    score=score,
                    label_text=label_text,
                    category_id=categories_dict[label_text],
                    xyxy=[float(v) for v in boxes[idx].tolist()],
                )
            )
        return detections


class SamSegmenter:
    def __init__(self, checkpoint_path: str, model_type: str = 'vit_h', device: str = 'cuda') -> None:
        self.predictor = load_sam_model(model_type=model_type, checkpoint=checkpoint_path, device=device)

    @staticmethod
    def _mask_to_coco_polygon(mask: np.ndarray) -> list[list[float]]:
        mask_uint8 = mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polygons: list[list[float]] = []
        for contour in contours:
            contour = contour.reshape(-1, 2)
            polygon = contour.flatten().astype(float).tolist()
            if len(polygon) >= 6:
                polygons.append(polygon)
        return polygons

    def segment(self, image_path: str, detections: Sequence[DetectionCandidate], image_id: int) -> list[dict[str, Any]]:
        if not detections:
            return []
        image = mmcv.imread(image_path, channel_order='rgb')
        image_np = np.array(image)
        results: list[dict[str, Any]] = []
        try:
            with torch.inference_mode():
                self.predictor.set_image(image_np)
                for det in detections:
                    xyxy = np.array(det.xyxy, dtype=np.float32)[None, :]
                    masks, _, _ = self.predictor.predict(
                        box=xyxy,
                        point_coords=None,
                        point_labels=None,
                        multimask_output=False,
                    )
                    x1, y1, x2, y2 = det.xyxy
                    results.append({
                        'image_id': image_id,
                        'score': det.score,
                        'category_id': det.category_id,
                        'bbox': [x1, y1, x2 - x1, y2 - y1],
                        'area': float((x2 - x1) * (y2 - y1)),
                        'segmentation': self._mask_to_coco_polygon(masks[0]),
                        'label_text': det.label_text,
                    })
        finally:
            if hasattr(self.predictor, 'reset_image'):
                self.predictor.reset_image()
        return results


class QwenGdinoSamProtocol:
    def __init__(self, controller: QwenController, localizer: GroundingDinoLocalizer, segmenter: SamSegmenter) -> None:
        self.controller = controller
        self.localizer = localizer
        self.segmenter = segmenter

    def run_query_only(
        self,
        query_image_path: str,
        image_id: int,
        categories_dict: dict[str, int],
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
        instruction: str | None = None,
    ) -> dict[str, Any]:
        controller_output = self.controller.infer_query_only(query_image_path, instruction=instruction)
        detections = self.localizer.detect(
            query_image_path,
            controller_output.cue_text,
            categories_dict,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
        segmented = self.segmenter.segment(query_image_path, detections, image_id=image_id)
        return {
            'image_id': image_id,
            'query_image_path': str(query_image_path),
            'controller_mode': 'query_only',
            'controller_instruction': controller_output.instruction,
            'controller_raw_text': controller_output.raw_text,
            'cue_text': controller_output.cue_text,
            'results': segmented,
        }

    def run_support_query(
        self,
        support_image_paths: Sequence[str],
        query_image_path: str,
        image_id: int,
        categories_dict: dict[str, int],
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
        instruction: str | None = None,
    ) -> dict[str, Any]:
        controller_output = self.controller.infer_support_query(
            support_image_paths,
            query_image_path,
            instruction=instruction,
        )
        detections = self.localizer.detect(
            query_image_path,
            controller_output.cue_text,
            categories_dict,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
        segmented = self.segmenter.segment(query_image_path, detections, image_id=image_id)
        return {
            'image_id': image_id,
            'query_image_path': str(query_image_path),
            'support_image_paths': [str(p) for p in support_image_paths],
            'controller_mode': 'support_query',
            'controller_instruction': controller_output.instruction,
            'controller_raw_text': controller_output.raw_text,
            'cue_text': controller_output.cue_text,
            'results': segmented,
        }


def load_categories_dict(dataset_json_path: str | Path) -> dict[str, int]:
    payload = json.loads(Path(dataset_json_path).read_text())
    return {cat['name'].replace('_', ' '): cat['id'] for cat in payload['categories']}


def load_support_image_paths(support_json_path: str | Path, support_dir: str | Path) -> list[str]:
    payload = json.loads(Path(support_json_path).read_text())
    support_dir = Path(support_dir)
    return [str((support_dir / image_info['file_name']).resolve()) for image_info in payload['images']]
