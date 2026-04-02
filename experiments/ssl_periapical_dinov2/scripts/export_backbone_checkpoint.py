#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoImageProcessor, AutoModel

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def main() -> None:
    parser = argparse.ArgumentParser(description="Exporta backbone do checkpoint SSL para formato HF")
    parser.add_argument("--checkpoint", required=True, help="checkpoint .pt salvo pelo treino SSL")
    parser.add_argument("--output-dir", required=True, help="diretorio de exportacao HF")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--backbone-key",
        default="student_backbone",
        choices=["student_backbone", "teacher_backbone"],
        help="qual backbone exportar do checkpoint (default: student_backbone)",
    )
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(ckpt_path, map_location=args.device)
    model_id = str(ckpt.get("model_id", "facebook/dinov2-small"))
    backbone_key = str(args.backbone_key)
    if backbone_key not in ckpt:
        avail = [k for k in ("student_backbone", "teacher_backbone") if k in ckpt]
        raise KeyError(f"Chave '{backbone_key}' nao encontrada no checkpoint. Disponiveis: {avail}")

    model = AutoModel.from_pretrained(model_id, local_files_only=False)
    model.load_state_dict(ckpt[backbone_key], strict=True)
    model.save_pretrained(out_dir)

    try:
        proc = AutoImageProcessor.from_pretrained(model_id, local_files_only=False)
        proc.save_pretrained(out_dir)
    except Exception:
        pass

    meta = {
        "source_checkpoint": str(ckpt_path.resolve()),
        "source_model_id": model_id,
        "backbone_key": backbone_key,
        "exported_backbone_dir": str(out_dir.resolve()),
        "epoch": int(ckpt.get("epoch", -1)),
        "global_step": int(ckpt.get("global_step", -1)),
    }
    (out_dir / "export_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
