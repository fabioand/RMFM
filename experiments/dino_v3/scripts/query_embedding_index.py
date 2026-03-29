#!/usr/bin/env python
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_distances

from dino_v3_lab import extract_global_embedding, load_backbone, resolve_device, resolve_hf_token


def main() -> None:
    parser = argparse.ArgumentParser(description="Consulta kNN por embedding para uma imagem")
    parser.add_argument("--model-id", default="facebook/dinov3-vits16-pretrain-lvd1689m")
    parser.add_argument("--index-dir", default="outputs/index")
    parser.add_argument("--query-image", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--hf-token-env", default="HF_TOKEN")
    parser.add_argument("--offline", action="store_true", help="Usa apenas cache local do Hugging Face")
    args = parser.parse_args()

    index_dir = Path(args.index_dir)
    matrix = np.load(index_dir / "embeddings.npy")
    paths = json.loads((index_dir / "paths.json").read_text(encoding="utf-8"))

    device = "cpu" if args.cpu else resolve_device()
    hf_token = resolve_hf_token(args.hf_token, args.hf_token_env)
    processor, model = load_backbone(
        args.model_id,
        device,
        hf_token=hf_token,
        local_files_only=args.offline,
    )
    q_img = Image.open(args.query_image).convert("RGB")
    q_emb = extract_global_embedding(processor, model, q_img, device)

    dists = cosine_distances(q_emb.reshape(1, -1), matrix).reshape(-1)
    order = np.argsort(dists)[: args.top_k]

    results = [{"path": paths[int(i)], "cosine_distance": float(dists[int(i)])} for i in order]
    print(json.dumps({"query_image": args.query_image, "top_k": args.top_k, "results": results}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
