from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Optional, Sequence

import torch

DEFAULT_INSTRUCTION = """
Task: Analyze the image and identify specific private objects, providing both category and location information.

16 categories to identify:
[local newspaper], [bank statement], [bills or receipt], [business card], [condom box],
[credit or debit card], [doctors prescription], [letters with address], [medical record document],
[pregnancy test], [empty pill bottle], [tattoo sleeve], [transcript], [mortgage or investment report],
[condom with plastic bag], [pregnancy test box]

Analysis steps:
1. Carefully examine the entire image to identify all potential objects
2. For each object, determine its category and approximate position (e.g., top-left, center, bottom-right)
3. Assess your confidence level for each classification (high, medium, low)

Identification guidelines:
- Empty pill bottle [empty pill bottle]: Cylindrical container, typically with white cap, translucent or opaque plastic material
- Condom with plastic bag [condom with plastic bag]: Small sealed transparent bag containing foil-wrapped item
- Bills or receipt [bills or receipt]: Rectangular paper with text blocks and numbers, typically neatly arranged
- Mortgage or investment report [mortgage or investment report]: Formal document with bold headers and financial data tables
- Transcript [transcript]: Multi-column academic-style document with dense text and numerical entries
- Tattoo sleeve [tattoo sleeve]: Colored fabric or sleeve, often with flame or tribal patterns
- Credit or debit card [credit or debit card]: Rectangular plastic card, metallic or colorful, with embedded logo/text
- Business card [business card]: Small rectangular card printed with contact information and logo
- Pregnancy test [pregnancy test]: Slim white plastic device with result window
- Pregnancy test box [pregnancy test box]: Vertical rectangular box with product branding and test device illustration
- Doctor's prescription [doctors prescription]: Medical form with structured layout and identification marks
- Condom box [condom box]: Small cardboard box, typically with commercial packaging design and small text
- Medical record document [medical record document]: Multi-page document with medical charts or diagrams
- Letters with address [letters with address]: Folded document with typed address block and formal formatting
- Local newspaper [local newspaper]: Full-page print layout with headlines, columns, and image thumbnails
- Bank statement [bank statement]: Document with transaction tables, charts, and bank logo formatting

Output format:
1. [category name] - Position: (describe position) - Confidence: (high/medium/low) - Features: (briefly describe identifying features)
2. [category name] - Position: (describe position) - Confidence: (high/medium/low) - Features: (briefly describe identifying features)
(continue listing if more objects are present...)
<output>category name,category name,category name</output>

If uncertain but possible categories exist, include them with low confidence. If no target categories can be identified in the image, respond with:
<output>No objects matching the given categories could be identified</output>
""".strip()


def release_torch_runtime() -> None:
    gc.collect()
    if not torch.cuda.is_available():
        return
    torch.cuda.empty_cache()
    if hasattr(torch.cuda, "ipc_collect"):
        torch.cuda.ipc_collect()


@dataclass
class DecodingConfig:
    temperature: float
    max_new_tokens: int
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    presence_penalty: Optional[float] = None
    seed: Optional[int] = None


def get_decoding_config(mode: str = "deterministic", seed: Optional[int] = None, **overrides) -> DecodingConfig:
    if mode == "deterministic":
        config = DecodingConfig(
            temperature=0.0,
            max_new_tokens=512,
            seed=None,
        )
    elif mode == "stochastic":
        config = DecodingConfig(
            temperature=0.7,
            max_new_tokens=512,
            top_p=0.8,
            top_k=20,
            presence_penalty=1.5,
            seed=42 if seed is None else seed,
        )
    else:
        raise ValueError("Unknown decoding mode: use 'deterministic' or 'stochastic'")

    for key, value in overrides.items():
        if value is not None and hasattr(config, key):
            setattr(config, key, value)
    return config


@dataclass
class SwiftVLMCaller:
    model_path: str
    max_new_tokens: int = 256
    decoding_mode: str = "deterministic"
    seed: Optional[int] = None
    max_pixels: int = 448
    device: Optional[str] = None
    lora_path: Optional[str] = None

    def __post_init__(self) -> None:
        config = get_decoding_config(
            self.decoding_mode,
            seed=self.seed,
            max_new_tokens=self.max_new_tokens,
        )
        self._use_legacy_swift = False

        try:
            from swift.llm import PtEngine, RequestConfig, safe_snapshot_download, get_model_tokenizer, get_template
            from swift.tuners import Swift
            self._use_legacy_swift = True
        except ModuleNotFoundError:
            try:
                from swift.infer_engine import InferRequest, RequestConfig, TransformersEngine
                from swift.utils import safe_snapshot_download
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    "MS-SWIFT is not installed. Install it first or use --llm-backend hf."
                ) from exc

        if config.temperature == 0.0:
            request_config = RequestConfig(
                max_tokens=config.max_new_tokens,
                temperature=0.0,
                top_k=1,
                top_p=1.0,
            )
        else:
            kwargs = {
                "max_tokens": config.max_new_tokens,
                "temperature": config.temperature,
            }
            if config.top_p is not None:
                kwargs["top_p"] = config.top_p
            if config.top_k is not None:
                kwargs["top_k"] = config.top_k
            if config.presence_penalty is not None:
                kwargs["repetition_penalty"] = config.presence_penalty
            if config.seed is not None:
                kwargs["seed"] = config.seed
            request_config = RequestConfig(**kwargs)

        self.request_config = request_config

        if self._use_legacy_swift:
            model, tokenizer = get_model_tokenizer(
                self.model_path,
                use_hf=True,
                max_pixels=self.max_pixels,
                device_map=_resolve_device_map(self.device),
            )
            if self.lora_path:
                lora_checkpoint = safe_snapshot_download(self.lora_path)
                model = Swift.from_pretrained(model, lora_checkpoint)
                model = model.merge_and_unload()
            model.eval()

            template = get_template(model.model_meta.template, tokenizer, default_system=None)
            self.engine = PtEngine.from_model_template(model, template, max_batch_size=1)
            self._InferRequest = __import__("swift.llm", fromlist=["InferRequest"]).InferRequest
            return

        adapters = None
        if self.lora_path:
            adapters = [safe_snapshot_download(self.lora_path, use_hf=True)]
        self.engine = TransformersEngine(
            self.model_path,
            adapters=adapters,
            max_batch_size=1,
            use_hf=True,
            max_pixels=self.max_pixels,
        )
        self._InferRequest = InferRequest

    def generate(self, image_path: str, instruction: Optional[str] = None) -> str:
        return self.generate_images([image_path], instruction=instruction)

    def generate_images(self, image_paths: Sequence[str], instruction: Optional[str] = None) -> str:
        if not image_paths:
            raise ValueError("image_paths must not be empty")
        prompt = instruction or DEFAULT_INSTRUCTION
        image_tokens = "".join("<image>" for _ in image_paths)
        infer_request = self._InferRequest(
            messages=[{"role": "user", "content": f"{image_tokens}{prompt}"}],
            images=[str(path) for path in image_paths],
        )
        with torch.inference_mode():
            resp_list = self.engine.infer([infer_request], self.request_config)
        text = resp_list[0].choices[0].message.content.strip()
        del resp_list
        del infer_request
        release_torch_runtime()
        return text


def _resolve_device_map(device: Optional[str]) -> Optional[object]:
    if not device:
        return None
    normalized = device.strip().lower()
    if normalized == "cpu":
        return "cpu"
    if normalized == "cuda":
        return {"": 0}
    if normalized.startswith("cuda:"):
        try:
            return {"": int(normalized.split(":", 1)[1])}
        except ValueError:
            return None
    return None
