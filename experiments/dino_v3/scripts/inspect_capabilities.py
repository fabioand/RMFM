#!/usr/bin/env python
import argparse
import json
import os
from pathlib import Path

from huggingface_hub import HfApi
from transformers import AutoConfig


def infer_capabilities(model_id: str, tags: list[str], architectures: list[str]) -> list[str]:
    caps = [
        "Feature extraction / embeddings (encoder backbone)",
        "Image retrieval via nearest neighbors",
        "Clustering and anomaly detection with embeddings",
        "Fine-tuning para classificação de imagens",
        "Fine-tuning para detecção/segmentação com heads adicionais",
    ]

    lowered_id = model_id.lower()
    if "dpt-head" in lowered_id or any("depth" in t.lower() for t in tags):
        caps.append("Depth estimation (checkpoint com head DPT)")

    if any("classification" in arch.lower() for arch in architectures):
        caps.append("Image classification out-of-the-box")

    return caps


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspeciona capacidades de um checkpoint DINOv3 no Hugging Face")
    parser.add_argument("--model-id", default="facebook/dinov3-vits16-pretrain-lvd1689m")
    parser.add_argument("--output", default="outputs/capabilities.json")
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--hf-token-env", default="HF_TOKEN")
    args = parser.parse_args()

    api = HfApi()
    hf_token = args.hf_token or os.environ.get(args.hf_token_env)
    info = api.model_info(args.model_id)
    architectures = []
    config_error = None
    try:
        cfg = AutoConfig.from_pretrained(args.model_id, token=hf_token)
        architectures = list(getattr(cfg, "architectures", []) or [])
    except Exception as exc:
        config_error = str(exc)

    tags = list(info.tags or [])
    capabilities = infer_capabilities(args.model_id, tags, architectures)

    payload = {
        "model_id": args.model_id,
        "library_name": info.library_name,
        "gated": info.gated,
        "pipeline_tag": info.pipeline_tag,
        "tags": tags,
        "architectures": architectures,
        "suggested_capabilities": capabilities,
        "config_error": config_error,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\nSalvo em: {output_path.resolve()}")


if __name__ == "__main__":
    main()
