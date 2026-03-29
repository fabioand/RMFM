#!/usr/bin/env python3
from __future__ import annotations

import argparse

from dino_periapical_cls.train_cached import run_training_cached


def main() -> None:
    parser = argparse.ArgumentParser(description="Treina head de classificação usando embeddings DINO pré-extraídos")
    parser.add_argument("--images-dir", required=True, help="Pasta de imagens periapicais")
    parser.add_argument("--labels-dir", required=True, help="Pasta de JSONs de classificação")
    parser.add_argument("--output-dir", required=True, help="Pasta de saída")

    parser.add_argument("--model-id", default="facebook/dinov2-small")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128, help="Batch do treino da cabeça")
    parser.add_argument("--feature-batch-size", type=int, default=64, help="Batch para pré-extração do encoder")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--shortest-edge", type=int, default=256)
    parser.add_argument("--crop-size", type=int, default=256)

    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=0, help="0 usa todas")

    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--hf-token-env", default="HF_TOKEN")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-tensorboard", action="store_true")
    parser.add_argument("--force-reextract-features", action="store_true", help="Ignora cache e reextrai features")
    parser.add_argument(
        "--augment-flip-mirror",
        action="store_true",
        help="Duplica apenas o treino com flip horizontal e remapeamento explícito de classe espelhada",
    )

    args = parser.parse_args()
    run_training_cached(args)


if __name__ == "__main__":
    main()
