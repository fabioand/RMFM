#!/usr/bin/env python
import argparse
import json
from pathlib import Path

import numpy as np
from huggingface_hub.errors import GatedRepoError

from dino_v3_lab import (
    extract_global_embedding,
    load_backbone,
    load_image,
    resolve_device,
    resolve_hf_token,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test DINOv3: carrega modelo, processa imagem e extrai embedding")
    parser.add_argument("--model-id", default="facebook/dinov3-vits16-pretrain-lvd1689m")
    parser.add_argument("--image", default=None, help="Caminho da imagem. Se omitido, usa imagem sintética.")
    parser.add_argument("--output", default="outputs/smoke_test_result.json")
    parser.add_argument("--cpu", action="store_true", help="Força execução em CPU")
    parser.add_argument("--hf-token", default=None, help="Token Hugging Face explícito (opcional)")
    parser.add_argument("--hf-token-env", default="HF_TOKEN", help="Variável de ambiente do token")
    parser.add_argument("--offline", action="store_true", help="Usa apenas cache local do Hugging Face")
    args = parser.parse_args()

    device = "cpu" if args.cpu else resolve_device()
    hf_token = resolve_hf_token(args.hf_token, args.hf_token_env)
    try:
        processor, model = load_backbone(
            args.model_id,
            device,
            hf_token=hf_token,
            local_files_only=args.offline,
        )
    except Exception as exc:
        msg = str(exc)
        if isinstance(exc, GatedRepoError) or "gated repo" in msg.lower():
            raise SystemExit(
                "Checkpoint gated. Solicite acesso no Hugging Face e rode novamente com "
                "--hf-token ou exporte HF_TOKEN."
            ) from exc
        raise
    image = load_image(args.image)
    emb = extract_global_embedding(processor, model, image, device)

    result = {
        "model_id": args.model_id,
        "device": device,
        "embedding_dim": int(emb.shape[0]),
        "embedding_l2_norm": float(np.linalg.norm(emb)),
        "embedding_preview": [float(x) for x in emb[:8]],
        "image_source": args.image if args.image else "synthetic",
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(result, indent=2))
    print(f"\nSalvo em: {output_path.resolve()}")


if __name__ == "__main__":
    main()
