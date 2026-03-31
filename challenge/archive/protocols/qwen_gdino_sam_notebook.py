from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from challenge.protocols.qwen_gdino_sam import (
    GroundingDinoLocalizer,
    QwenController,
    QwenGdinoSamProtocol,
    SamSegmenter,
    load_categories_dict,
)


class ProtocolSession:
    def __init__(
        self,
        llm_model: str,
        gdino_config: str,
        gdino_checkpoint: str,
        sam_checkpoint: str,
        dataset_json: str,
        device: str = 'cuda',
        llm_max_new_tokens: int = 256,
        llm_decoding_mode: str = 'deterministic',
        llm_seed: int | None = None,
        llm_max_pixels: int = 448,
        query_instruction: str | None = None,
        support_query_instruction: str | None = None,
    ) -> None:
        self.categories_dict = load_categories_dict(dataset_json)
        controller = QwenController(
            model_path=llm_model,
            max_new_tokens=llm_max_new_tokens,
            decoding_mode=llm_decoding_mode,
            seed=llm_seed,
            max_pixels=llm_max_pixels,
            instruction=query_instruction,
            support_query_instruction=support_query_instruction,
            allowed_categories=list(self.categories_dict.keys()),
        )
        localizer = GroundingDinoLocalizer(gdino_config, gdino_checkpoint, device=device)
        segmenter = SamSegmenter(sam_checkpoint, device=device)
        self.protocol = QwenGdinoSamProtocol(controller, localizer, segmenter)
        self.query_instruction = controller.get_default_query_instruction()
        self.support_query_instruction = controller.get_default_support_query_instruction()

    def show_default_prompts(self) -> None:
        print('=== query_only default prompt ===')
        print(self.query_instruction)
        print('\n=== support_query default prompt ===')
        print(self.support_query_instruction)

    def run_query_only(
        self,
        query_image_path: str,
        image_id: int = 0,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
        instruction: str | None = None,
    ):
        return self.protocol.run_query_only(
            query_image_path=query_image_path,
            image_id=image_id,
            categories_dict=self.categories_dict,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            instruction=instruction,
        )

    def run_support_query(
        self,
        support_image_paths: Sequence[str],
        query_image_path: str,
        image_id: int = 0,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
        instruction: str | None = None,
    ):
        return self.protocol.run_support_query(
            support_image_paths=support_image_paths,
            query_image_path=query_image_path,
            image_id=image_id,
            categories_dict=self.categories_dict,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            instruction=instruction,
        )


def show_protocol_result(record: dict, figsize=(10, 10)) -> None:
    image = Image.open(record['query_image_path']).convert('RGB')
    draw = ImageDraw.Draw(image)
    for result in record.get('results', []):
        x, y, w, h = result['bbox']
        draw.rectangle([x, y, x + w, y + h], outline='red', width=4)
        draw.text((x, max(0, y - 18)), f"{result['label_text']} {result['score']:.2f}", fill='red')
    plt.figure(figsize=figsize)
    plt.imshow(image)
    plt.axis('off')
    plt.show()


def print_protocol_summary(record: dict) -> None:
    print('controller_mode:', record.get('controller_mode'))
    if 'support_image_paths' in record:
        print('num_support_images:', len(record['support_image_paths']))
    print('controller_instruction:')
    print(record.get('controller_instruction', ''))
    print('cue_text:', record.get('cue_text'))
    print('controller_raw_text:')
    print(record.get('controller_raw_text', ''))
    print('num_results:', len(record.get('results', [])))
    for idx, result in enumerate(record.get('results', []), start=1):
        print(f"  [{idx}] label={result['label_text']} score={result['score']:.3f} bbox={result['bbox']}")


def resolve_images(image_dir: str, *file_names: str) -> list[str]:
    base = Path(image_dir)
    return [str((base / name).resolve()) for name in file_names]