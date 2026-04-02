#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from transformers import AutoImageProcessor

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ssl_periapical_dinov2 import SSLMultiCropTransform, list_images_from_source, load_config


def _panel_h(images, titles, out_path: Path) -> None:
    if not images:
        return
    h, w, _ = images[0].shape
    bar_h = 24
    panel = np.zeros((h + bar_h, w * len(images), 3), dtype=np.uint8)
    for i, (img, title) in enumerate(zip(images, titles)):
        x0 = i * w
        panel[bar_h:, x0 : x0 + w] = img
    pil = Image.fromarray(panel)
    draw = ImageDraw.Draw(pil)
    for i, title in enumerate(titles):
        draw.text((i * w + 5, 5), title, fill=(230, 230, 230))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pil.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview do multicrop/augmentacao SSL em amostras")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ds = cfg["dataset"]
    mc = cfg["multicrop"]
    aug = cfg["augmentation"]
    model_id = str(cfg["model"]["model_id"])
    processor = AutoImageProcessor.from_pretrained(model_id, local_files_only=bool(cfg["model"].get("offline", False)))
    transform = SSLMultiCropTransform(
        global_crops=int(mc["global_crops"]),
        local_crops=int(mc["local_crops"]),
        global_size=int(mc["global_size"]),
        local_size=int(mc["local_size"]),
        global_scale=tuple(float(x) for x in mc["global_scale"]),
        local_scale=tuple(float(x) for x in mc["local_scale"]),
        ratio=tuple(float(x) for x in mc["ratio"]),
        rotation_deg=float(aug["rotation_deg"]),
        brightness_delta=float(aug["brightness_delta"]),
        contrast_delta=float(aug["contrast_delta"]),
        blur_prob=float(aug["blur_prob"]),
        noise_prob=float(aug["noise_prob"]),
        noise_std=float(aug["noise_std"]),
        mean=tuple(float(x) for x in getattr(processor, "image_mean", [0.485, 0.456, 0.406])),
        std=tuple(float(x) for x in getattr(processor, "image_std", [0.229, 0.224, 0.225])),
    )

    paths = list_images_from_source(
        images_dir=str(ds.get("images_dir", "")),
        list_txt=str(ds.get("list_txt", "")),
        list_json=str(ds.get("list_json", "")),
        recursive=bool(ds.get("recursive", False)),
    )
    rng = random.Random(int(args.seed))
    rng.shuffle(paths)
    n = min(int(args.num_samples), len(paths))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, p in enumerate(paths[:n]):
        base = Image.open(p)
        g = ImageOps.grayscale(base)
        rgb = Image.merge("RGB", (g, g, g))
        b = transform.sample_views(rgb)
        metas = b["meta"]
        titles = [f"{m['kind']} a={m['area_scale']:.3f} r={m['crop_ratio']:.2f}" for m in metas]
        _panel_h(b["preview_images"], titles, out_dir / f"{i:02d}_{p.stem}.png")

    print(f"[PREVIEW] samples={n}")
    print(f"[PREVIEW] output={out_dir}")


if __name__ == "__main__":
    main()
