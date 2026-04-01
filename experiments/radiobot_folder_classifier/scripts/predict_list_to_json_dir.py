#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image, UnidentifiedImageError
from torch import nn
from transformers import AutoImageProcessor

# Allow running from repo root or from inside the experiment folder.
_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from dino_folder_cls.data import Sample, VALID_IMAGE_EXTS, discover_samples_from_list_json
from dino_folder_cls.model import FrozenDinoClassifier


class HeadClassifier(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(dropout),
            nn.Linear(in_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def resolve_device(prefer_mps: bool = True) -> str:
    if torch.cuda.is_available():
        return "cuda"
    if prefer_mps and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _safe_json_name(stem: str, full_path: str) -> str:
    # Keep stem.json by default; append short hash only for collisions.
    short = hashlib.sha1(full_path.encode("utf-8")).hexdigest()[:8]
    return f"{stem}__{short}.json"


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _discover_samples_from_images_dir(images_dir: Path, recursive: bool = False) -> list[Sample]:
    root = images_dir.resolve()
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"--images-dir inválido: {images_dir}")

    it = root.rglob("*") if recursive else root.glob("*")
    samples: list[Sample] = []
    for p in sorted(it):
        if not p.is_file():
            continue
        if p.suffix.lower() not in VALID_IMAGE_EXTS:
            continue
        if p.name.startswith("._") or "__MACOSX" in p.parts:
            continue
        try:
            rel_parent = p.parent.resolve().relative_to(root)
            label_name = "." if str(rel_parent) == "." else str(rel_parent)
        except Exception:
            label_name = p.parent.name
        samples.append(Sample(image_path=p.resolve(), stem=p.stem, label_name=label_name))
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classifica imagens de um list-json e salva um JSON por imagem (pred + confiança + softmax por classe)."
    )
    parser.add_argument("--run-dir", required=True, help="Diretório do treino (contém best_head_only.pt e cache_meta).")
    parser.add_argument("--list-json", default="", help="Lista JSON de imagens (sample_list).")
    parser.add_argument("--images-dir", default="", help="Pasta de imagens para inferência direta (sem list-json).")
    parser.add_argument("--recursive", action="store_true", help="Com --images-dir, percorre recursivamente.")
    parser.add_argument("--output-dir", required=True, help="Pasta de saída para os JSONs por imagem.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=5, help="Top-K classes para salvar em lista ordenada.")
    parser.add_argument("--shortest-edge", type=int, default=-1, help="Override do preprocess. -1 usa valor do treino.")
    parser.add_argument("--crop-size", type=int, default=-1, help="Override do preprocess. -1 usa valor do treino.")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--progress-every", type=int, default=200)
    args = parser.parse_args()

    print("[1/6] Validando run/checkpoint...")
    run_dir = Path(args.run_dir).resolve()
    ckpt_path = run_dir / "best_head_only.pt"
    cache_meta_path = run_dir / "features_cache" / "cache_meta.json"
    if not ckpt_path.exists():
        raise SystemExit(f"Checkpoint não encontrado: {ckpt_path}")
    if not cache_meta_path.exists():
        raise SystemExit(f"cache_meta.json não encontrado: {cache_meta_path}")

    if bool(args.list_json) == bool(args.images_dir):
        raise SystemExit("Use exatamente um: --list-json OU --images-dir.")

    print("[2/6] Carregando lista de imagens...")
    if args.list_json:
        samples = discover_samples_from_list_json(Path(args.list_json).resolve())
        input_desc = str(Path(args.list_json).resolve())
        input_mode = "list_json"
    else:
        samples = _discover_samples_from_images_dir(Path(args.images_dir).resolve(), recursive=bool(args.recursive))
        input_desc = str(Path(args.images_dir).resolve())
        input_mode = "images_dir"

    if not samples:
        raise SystemExit("Nenhuma imagem válida encontrada na entrada.")
    print(f"       amostras válidas: {len(samples)}")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[3/6] Carregando processor/model/checkpoint...")
    cache_meta = json.loads(cache_meta_path.read_text(encoding="utf-8"))
    model_id = str(cache_meta["model_id"])
    shortest_edge = int(cache_meta["shortest_edge"]) if int(args.shortest_edge) <= 0 else int(args.shortest_edge)
    crop_size = int(cache_meta["crop_size"]) if int(args.crop_size) <= 0 else int(args.crop_size)

    device = "cpu" if bool(args.cpu) else resolve_device()
    ckpt = torch.load(ckpt_path, map_location=device)
    label_to_idx = {str(k): int(v) for k, v in ckpt["label_to_idx"].items()}
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    in_dim = int(ckpt["in_dim"])
    dropout = float(ckpt.get("args", {}).get("dropout", 0.1))

    processor = AutoImageProcessor.from_pretrained(
        model_id,
        local_files_only=bool(args.offline),
    )
    backbone = FrozenDinoClassifier(
        model_id=model_id,
        num_classes=len(label_to_idx),
        local_files_only=bool(args.offline),
        freeze_backbone=True,
    ).to(device)
    backbone.eval()
    head = HeadClassifier(in_dim=in_dim, num_classes=len(label_to_idx), dropout=dropout).to(device)
    head.load_state_dict(ckpt["head_state_dict"])
    head.eval()
    print(f"       device: {device} | classes: {len(label_to_idx)} | batch: {args.batch_size}")

    top_k = max(1, min(int(args.top_k), len(label_to_idx)))
    progress_every = max(1, int(args.progress_every))
    batch_size = max(1, int(args.batch_size))
    next_progress_mark = progress_every

    collisions = 0
    written = 0
    skipped_invalid = 0
    skipped_runtime = 0
    t0 = time.perf_counter()
    pred_counts: dict[str, int] = {}
    saved_names: set[str] = set()
    errors_jsonl = output_dir / "_errors.jsonl"
    if errors_jsonl.exists():
        errors_jsonl.unlink()

    print("[4/6] Inferência em lote + escrita de JSONs...")
    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            chunk = samples[start : start + batch_size]
            images: list[Image.Image] = []
            valid_chunk: list[Sample] = []
            for s in chunk:
                try:
                    images.append(Image.open(s.image_path).convert("RGB"))
                    valid_chunk.append(s)
                except (UnidentifiedImageError, OSError, ValueError) as e:
                    skipped_invalid += 1
                    _append_jsonl(
                        errors_jsonl,
                        {
                            "type": "invalid_image",
                            "image_path": str(s.image_path),
                            "stem": s.stem,
                            "error": str(e),
                        },
                    )
            if not valid_chunk:
                done = start + len(chunk)
                if done >= next_progress_mark:
                    print(f"{done}/{len(samples)}")
                    while done >= next_progress_mark:
                        next_progress_mark += progress_every
                if done >= len(samples):
                    print(f"{done}/{len(samples)}")
                continue

            proc_kwargs: dict[str, Any] = {}
            if shortest_edge > 0:
                proc_kwargs["size"] = {"shortest_edge": shortest_edge}
            if crop_size > 0:
                proc_kwargs["crop_size"] = {"height": crop_size, "width": crop_size}

            try:
                inputs = processor(images=images, return_tensors="pt", **proc_kwargs)
                pixel_values = inputs["pixel_values"].to(device)

                feats = backbone._extract_features(pixel_values)
                logits = head(feats)
                probs = torch.softmax(logits, dim=1).detach().cpu()

                for i, s in enumerate(valid_chunk):
                    row_probs = probs[i]
                    conf, pred = torch.max(row_probs, dim=0)
                    pred_idx = int(pred.item())
                    pred_label = idx_to_label[pred_idx]
                    pred_counts[pred_label] = pred_counts.get(pred_label, 0) + 1

                    top_p, top_i = torch.topk(row_probs, k=top_k)
                    top_classes = [
                        {
                            "label": idx_to_label[int(idx.item())],
                            "prob": float(p.item()),
                        }
                        for p, idx in zip(top_p, top_i)
                    ]
                    probs_by_class = {
                        idx_to_label[j]: float(row_probs[j].item()) for j in range(len(label_to_idx))
                    }

                    json_name = f"{s.stem}.json"
                    out_path = output_dir / json_name
                    if json_name in saved_names or out_path.exists():
                        collisions += 1
                        out_path = output_dir / _safe_json_name(s.stem, str(s.image_path.resolve()))
                    saved_names.add(out_path.name)

                    payload = {
                        "image_path": str(s.image_path.resolve()),
                        "stem": s.stem,
                        "source_folder": s.label_name,
                        "pred_idx": pred_idx,
                        "pred_label": pred_label,
                        "pred_confidence": float(conf.item()),
                        "top_classes": top_classes,
                        "probs_by_class": probs_by_class,
                        "model": {
                            "model_id": model_id,
                            "run_dir": str(run_dir),
                            "shortest_edge": shortest_edge,
                            "crop_size": crop_size,
                        },
                    }
                    _save_json(out_path, payload)
                    written += 1
            except Exception as e:
                skipped_runtime += len(valid_chunk)
                _append_jsonl(
                    errors_jsonl,
                    {
                        "type": "batch_runtime_error",
                        "range_start": start,
                        "range_end": start + len(chunk) - 1,
                        "num_images": len(valid_chunk),
                        "error": str(e),
                    },
                )

            done = start + len(chunk)
            printed_now = False
            if done >= next_progress_mark:
                pct = (100.0 * done) / max(1, len(samples))
                print(f"       progresso: {done}/{len(samples)} ({pct:.1f}%)")
                printed_now = True
                while done >= next_progress_mark:
                    next_progress_mark += progress_every
            if done >= len(samples) and not printed_now:
                print(f"       progresso: {done}/{len(samples)} (100.0%)")

    t1 = time.perf_counter()
    print("[5/6] Gravando resumo...")
    summary = {
        "mode": "predict_list_to_json_dir",
        "run_dir": str(run_dir),
        "input_mode": input_mode,
        "input": input_desc,
        "device": device,
        "num_input_samples": len(samples),
        "num_json_written": written,
        "json_name_collisions": collisions,
        "num_skipped_invalid_image": skipped_invalid,
        "num_skipped_runtime_error": skipped_runtime,
        "errors_jsonl": str(errors_jsonl),
        "pred_counts": dict(sorted(pred_counts.items(), key=lambda kv: kv[0])),
        "elapsed_s": float(t1 - t0),
        "mean_images_per_s": float(written / max(1e-9, (t1 - t0))),
        "output_dir": str(output_dir),
    }
    _save_json(output_dir / "_summary.json", summary)
    print("[6/6] Concluído.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
