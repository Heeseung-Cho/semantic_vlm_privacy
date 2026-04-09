from contextlib import contextmanager

import torch
import os
from mmdet.apis import DetInferencer
from segment_anything import SamPredictor, sam_model_registry
from PIL import ImageOps, Image


@contextmanager
def _trusted_torch_load_context():
    original_torch_load = torch.load

    def patched_torch_load(*args, **kwargs):
        kwargs.setdefault('weights_only', False)
        return original_torch_load(*args, **kwargs)

    torch.load = patched_torch_load
    try:
        yield
    finally:
        torch.load = original_torch_load


def _patch_glip_runtime():
    try:
        import nltk
        from mmdet.models.detectors import glip as glip_module
    except ImportError:
        return

    if getattr(glip_module, "_challenge_repo_nltk_patched", False):
        return

    nltk_data_dir = os.path.expanduser("~/nltk_data")
    os.makedirs(nltk_data_dir, exist_ok=True)

    def _ensure_nltk_resource(path: str, resource: str) -> None:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(resource, download_dir=nltk_data_dir, quiet=True)

    def _find_noun_phrases_once(caption: str) -> list:
        _ensure_nltk_resource("tokenizers/punkt", "punkt")
        _ensure_nltk_resource(
            "taggers/averaged_perceptron_tagger",
            "averaged_perceptron_tagger",
        )

        caption = caption.lower()
        tokens = nltk.word_tokenize(caption)
        pos_tags = nltk.pos_tag(tokens)

        grammar = "NP: {<DT>?<JJ.*>*<NN.*>+}"
        cp = nltk.RegexpParser(grammar)
        result = cp.parse(pos_tags)

        noun_phrases = []
        for subtree in result.subtrees():
            if subtree.label() == "NP":
                noun_phrases.append(" ".join(t[0] for t in subtree.leaves()))
        return noun_phrases

    def _run_ner_quiet(caption: str):
        noun_phrases = _find_noun_phrases_once(caption)
        noun_phrases = [glip_module.remove_punctuation(phrase) for phrase in noun_phrases]
        noun_phrases = [phrase for phrase in noun_phrases if phrase != ""]
        relevant_phrases = noun_phrases
        labels = noun_phrases

        tokens_positive = []
        for entity, label in zip(relevant_phrases, labels):
            try:
                for m in glip_module.re.finditer(entity, caption.lower()):
                    tokens_positive.append([[m.start(), m.end()]])
            except Exception:
                print("noun entities:", noun_phrases)
                print("entity:", entity)
                print("caption:", caption.lower())
        return tokens_positive, noun_phrases

    glip_module.find_noun_phrases = _find_noun_phrases_once
    glip_module.run_ner = _run_ner_quiet
    glip_module._challenge_repo_nltk_patched = True


def load_sam_model(model_type="vit_h", checkpoint="sam_vit_h_4b8939.pth", device="cuda"):
    """
    Load Segment Anything Model (SAM)
    """
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    resolved_device = device
    if resolved_device == "cuda" and not torch.cuda.is_available():
        resolved_device = "cpu"
        print("CUDA not available, using CPU instead")
    sam.to(device=resolved_device)
    predictor = SamPredictor(sam)
    return predictor

def load_groundingdino_model(config_path="./configs/grounding_dino_swin-t_finetune_8xb2_20e_o365.py", checkpoint_path="./checkpoints/groundingdino_swint_ogc.pth", device="cuda"):
    """
    Load MM Grounding DINO model from mmdet
    """
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("CUDA not available, using CPU instead")

    _patch_glip_runtime()

    with _trusted_torch_load_context():
        model = DetInferencer(
            model=config_path,
            weights=checkpoint_path,
            device=device
        )

    return model

def load_clip_model(device="cuda", clip_model_path=None):
    """
    Load fine-tuned OpenCLIP model
    """
    import open_clip
    model_name = "ViT-B-16"
    default_local_pretrained = os.path.join(
        os.path.dirname(__file__),
        "checkpoints",
        "ViT-B-16.pt",
    )
    pretrained = clip_model_path or (default_local_pretrained if os.path.exists(default_local_pretrained) else "openai")
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name=model_name,
        pretrained=pretrained,
        load_weights_only=False,
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    model.to(device)
    return model, preprocess, tokenizer
