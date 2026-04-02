#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ssl_periapical_dinov2 import load_config, run_ssl_training


def _merge_overrides(cfg: dict, args) -> dict:
    out = json.loads(json.dumps(cfg))
    if args.run_name:
        out["run"]["name"] = args.run_name
    if args.output_dir:
        out["run"]["output_dir"] = args.output_dir
    if args.images_dir:
        out["dataset"]["images_dir"] = args.images_dir
        out["dataset"]["list_txt"] = ""
        out["dataset"]["list_json"] = ""
    if args.list_txt:
        out["dataset"]["list_txt"] = args.list_txt
        out["dataset"]["images_dir"] = ""
        out["dataset"]["list_json"] = ""
    if args.list_json:
        out["dataset"]["list_json"] = args.list_json
        out["dataset"]["images_dir"] = ""
        out["dataset"]["list_txt"] = ""
    if args.max_images > 0:
        out["dataset"]["max_images"] = int(args.max_images)
    if args.epochs > 0:
        out["training"]["epochs"] = int(args.epochs)
    if args.batch_size > 0:
        out["training"]["batch_size"] = int(args.batch_size)
    if args.cpu:
        out["training"]["cpu"] = True
    if args.offline:
        out["model"]["offline"] = True
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Treino SSL DINOv2 (teacher-student + multicrop)")
    parser.add_argument("--config", required=True, help="Arquivo de config .yaml/.json")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--images-dir", default="")
    parser.add_argument("--list-txt", default="")
    parser.add_argument("--list-json", default="")
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = _merge_overrides(cfg, args)
    run_ssl_training(cfg)


if __name__ == "__main__":
    main()

