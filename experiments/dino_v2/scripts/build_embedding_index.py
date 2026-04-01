#!/usr/bin/env python
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.neighbors import NearestNeighbors

from dino_v3_lab import extract_global_embedding, load_backbone, resolve_device, resolve_hf_token

VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def iter_images(folder: Path):
    for p in sorted(folder.rglob("*")):
        if p.is_file() and p.suffix.lower() in VALID_EXT:
            yield p


def main() -> None:
    parser = argparse.ArgumentParser(description="Extrai embeddings e cria índice kNN para retrieval")
    parser.add_argument("--model-id", default="facebook/dinov3-vits16-pretrain-lvd1689m")
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--output-dir", default="outputs/index")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--hf-token-env", default="HF_TOKEN")
    parser.add_argument("--offline", action="store_true", help="Usa apenas cache local do Hugging Face")
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = list(iter_images(images_dir))
    if not image_paths:
        raise RuntimeError(f"Nenhuma imagem encontrada em {images_dir}")

    device = "cpu" if args.cpu else resolve_device()
    hf_token = resolve_hf_token(args.hf_token, args.hf_token_env)
    processor, model = load_backbone(
        args.model_id,
        device,
        hf_token=hf_token,
        local_files_only=args.offline,
    )

    embeddings = []
    rel_paths = []
    for p in image_paths:
        img = Image.open(p).convert("RGB")
        emb = extract_global_embedding(processor, model, img, device)
        embeddings.append(emb)
        rel_paths.append(str(p.relative_to(images_dir)))

    matrix = np.stack(embeddings, axis=0)
    index = NearestNeighbors(n_neighbors=min(10, len(rel_paths)), metric="cosine")
    index.fit(matrix)

    np.save(output_dir / "embeddings.npy", matrix)
    (output_dir / "paths.json").write_text(json.dumps(rel_paths, indent=2, ensure_ascii=False), encoding="utf-8")

    meta = {
        "model_id": args.model_id,
        "device": device,
        "num_images": len(rel_paths),
        "embedding_dim": int(matrix.shape[1]),
        "index_type": "sklearn.neighbors.NearestNeighbors(metric=cosine)",
    }
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"\nÍndice salvo em: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
