from __future__ import annotations

import math
import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import AutoImageProcessor, AutoModel


def resolve_device(prefer_mps: bool = True) -> str:
    if torch.cuda.is_available():
        return "cuda"
    if prefer_mps and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_hf_token(explicit_token: Optional[str] = None, token_env: str = "HF_TOKEN") -> Optional[str]:
    if explicit_token:
        return explicit_token
    return os.environ.get(token_env)


def load_backbone(
    model_id: str,
    device: str,
    hf_token: Optional[str] = None,
    local_files_only: bool = False,
) -> Tuple[AutoImageProcessor, AutoModel]:
    processor = AutoImageProcessor.from_pretrained(
        model_id,
        token=hf_token,
        local_files_only=local_files_only,
    )
    model = AutoModel.from_pretrained(
        model_id,
        token=hf_token,
        local_files_only=local_files_only,
    )
    model.eval()
    model.to(device)
    return processor, model


def load_image(image_path: Optional[str] = None, image_size: int = 518) -> Image.Image:
    if image_path:
        return Image.open(image_path).convert("RGB")

    # Synthetic image for deterministic smoke tests when no sample is provided.
    img = Image.new("RGB", (image_size, image_size), color=(20, 20, 20))
    draw = ImageDraw.Draw(img)
    draw.ellipse((80, 80, image_size - 80, image_size - 80), outline=(220, 220, 220), width=6)
    draw.rectangle(
        (image_size // 3, image_size // 3, image_size // 3 + 90, image_size // 3 + 60),
        fill=(180, 180, 180),
    )
    return img


def extract_global_embedding(
    processor,
    model,
    image: Image.Image,
    device: str,
    processor_kwargs: Optional[Dict] = None,
) -> np.ndarray:
    with torch.no_grad():
        kwargs = processor_kwargs or {}
        inputs = processor(images=image, return_tensors="pt", **kwargs)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)

        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            emb = outputs.pooler_output[0]
        elif hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            emb = outputs.last_hidden_state[0].mean(dim=0)
        else:
            raise RuntimeError("Model output does not have pooler_output or last_hidden_state.")

        emb = torch.nn.functional.normalize(emb, dim=0)
    return emb.detach().cpu().numpy()


def extract_global_embedding_and_cls_patch_map(
    processor,
    model,
    image: Image.Image,
    device: str,
    processor_kwargs: Optional[Dict] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        kwargs = processor_kwargs or {}
        inputs = processor(images=image, return_tensors="pt", **kwargs)
        pixel_values = inputs["pixel_values"]
        h_px = int(pixel_values.shape[-2])
        w_px = int(pixel_values.shape[-1])

        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)

        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            emb = outputs.pooler_output[0]
        elif hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            emb = outputs.last_hidden_state[0].mean(dim=0)
        else:
            raise RuntimeError("Model output does not have pooler_output or last_hidden_state.")
        emb = torch.nn.functional.normalize(emb, dim=0)

        if not hasattr(outputs, "last_hidden_state") or outputs.last_hidden_state is None:
            raise RuntimeError("Model output does not have last_hidden_state for attention map.")

        hidden = outputs.last_hidden_state[0]  # [1 + N, D]
        if hidden.shape[0] < 2:
            raise RuntimeError("last_hidden_state too short to compute cls-patch map.")

        cls_tok = hidden[0]
        patch_toks = hidden[1:]
        cls_tok = torch.nn.functional.normalize(cls_tok, dim=0)
        patch_toks = torch.nn.functional.normalize(patch_toks, dim=1)
        scores = torch.sum(patch_toks * cls_tok.unsqueeze(0), dim=1)  # [N]

        patch_count = int(scores.shape[0])
        patch_size = int(getattr(model.config, "patch_size", 0) or 0)

        h_tok = 0
        w_tok = 0
        if patch_size > 0 and h_px % patch_size == 0 and w_px % patch_size == 0:
            h_tok = h_px // patch_size
            w_tok = w_px // patch_size
            if h_tok * w_tok != patch_count:
                h_tok = 0
                w_tok = 0
        if h_tok == 0 or w_tok == 0:
            side = int(math.sqrt(patch_count))
            if side * side != patch_count:
                raise RuntimeError(f"Cannot reshape {patch_count} patch tokens into 2D map.")
            h_tok = side
            w_tok = side

        map_tokens = scores.reshape(1, 1, h_tok, w_tok)
        map_full = torch.nn.functional.interpolate(
            map_tokens,
            size=(h_px, w_px),
            mode="bilinear",
            align_corners=False,
        )[0, 0]

        mn = torch.min(map_full)
        mx = torch.max(map_full)
        if float(mx - mn) > 1e-12:
            map_full = (map_full - mn) / (mx - mn)
        else:
            map_full = torch.zeros_like(map_full)

    return emb.detach().cpu().numpy(), map_full.detach().cpu().numpy()
