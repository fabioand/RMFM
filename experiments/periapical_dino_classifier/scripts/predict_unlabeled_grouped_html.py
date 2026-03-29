#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn
from transformers import AutoImageProcessor

from dino_periapical_cls.data import VALID_IMAGE_EXTS
from dino_periapical_cls.model import FrozenDinoClassifier


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


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _collect_image_paths(images_dir: Path) -> list[Path]:
    out: list[Path] = []
    for p in sorted(images_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTS:
            out.append(p)
    return out


def _collect_gt_stems_from_labels_dir(labels_dir: Path) -> set[str]:
    return {p.stem for p in labels_dir.glob("*.json") if p.is_file()}


def _collect_gt_stems_from_images_dir(images_dir: Path) -> set[str]:
    stems: set[str] = set()
    for p in images_dir.iterdir():
        if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTS:
            stems.add(p.stem)
    return stems


def build_grouped_html(title: str, rows: list[dict[str, Any]], idx_to_label: dict[int, str], summary: dict[str, Any]) -> str:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        grouped[int(r["pred_idx"])].append(r)

    sections = []
    for pred_idx in sorted(grouped.keys()):
        items = grouped[pred_idx]
        cls_name = idx_to_label[pred_idx]
        cards = []
        for item in sorted(items, key=lambda x: x["confidence"], reverse=True):
            img_uri = Path(item["image"]).resolve().as_uri()
            cards.append(
                f"""
                <article class="card">
                  <div class="thumb"><img src="{img_uri}" alt="{html.escape(item['stem'])}" loading="lazy" /></div>
                  <div class="body">
                    <h3 title="{html.escape(item['image'])}">{html.escape(item['stem'])}</h3>
                    <p><b>Pred:</b> {html.escape(item['pred_label'])}</p>
                    <p><b>Conf:</b> {item['confidence']:.3f}</p>
                    <p><b>Inferência:</b> {item['inference_ms']:.2f} ms</p>
                    <p><b>Preprocess:</b> {item['preprocess_ms']:.2f} ms</p>
                    <p><b>Total:</b> {item['total_ms']:.2f} ms</p>
                  </div>
                </article>
                """
            )

        sections.append(
            f"""
            <section class="cluster">
              <h2>{html.escape(cls_name)} (pred) - {len(items)} imgs</h2>
              <div class="grid">{''.join(cards)}</div>
            </section>
            """
        )

    return f"""<!doctype html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      --bg: #f4f7fb;
      --ink: #111827;
      --muted: #4b5563;
      --card: #fff;
      --line: #d1d9e6;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: var(--ink); background: var(--bg); }}
    .wrap {{ max-width: 1540px; margin: 0 auto; padding: 24px; }}
    h1 {{ margin: 0 0 8px; font-size: 24px; }}
    .meta {{ margin: 0 0 16px; font-size: 13px; color: var(--muted); }}
    .cluster {{ margin-top: 18px; }}
    h2 {{ margin: 0 0 10px; font-size: 17px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(230px, 1fr)); gap: 12px; }}
    .card {{ border: 1px solid var(--line); border-radius: 12px; background: var(--card); overflow: hidden; }}
    .thumb {{ height: 170px; background: #eef2f7; }}
    .thumb img {{ width: 100%; height: 100%; object-fit: contain; }}
    .body {{ padding: 10px 11px 12px; }}
    .body h3 {{ margin: 0 0 8px; font-size: 12px; line-height: 1.35; word-break: break-all; }}
    .body p {{ margin: 4px 0; font-size: 12px; color: var(--muted); }}
  </style>
</head>
<body>
  <main class="wrap">
    <h1>{html.escape(title)}</h1>
    <p class="meta">
      imgs={summary['num_images']} | device={summary['device']} |
      preprocess_mean={summary['timing']['preprocess_mean_ms']:.2f}ms |
      infer_mean={summary['timing']['inference_mean_ms']:.2f}ms |
      total_mean={summary['timing']['total_mean_ms']:.2f}ms
    </p>
    {''.join(sections)}
  </main>
</body>
</html>"""


def _timing_stats(values_ms: list[float]) -> dict[str, float]:
    arr = np.array(values_ms, dtype=np.float64)
    if arr.size == 0:
        return {
            "mean_ms": 0.0,
            "median_ms": 0.0,
            "p95_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
        }
    return {
        "mean_ms": float(arr.mean()),
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classifica imagens sem GT com ViT+head treinado e gera HTML agrupado por classe predita."
    )
    parser.add_argument("--run-dir", required=True, help="Pasta do run com best_head_only.pt e features_cache/cache_meta.json")
    parser.add_argument("--images-dir", required=True, help="Pasta com imagens candidatas (ex.: periapicais_3000)")
    parser.add_argument("--exclude-labels-dir", default="", help="Pasta com .json de GT para excluir por stem")
    parser.add_argument("--exclude-images-dir", default="", help="Pasta com imagens GT para excluir por stem")
    parser.add_argument("--num-images", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="", help="Default: <run-dir>/predict_unlabeled_600")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    ckpt_path = run_dir / "best_head_only.pt"
    cache_meta_path = run_dir / "features_cache" / "cache_meta.json"
    if not ckpt_path.exists():
        raise SystemExit(f"Checkpoint não encontrado: {ckpt_path}")
    if not cache_meta_path.exists():
        raise SystemExit(f"cache_meta.json não encontrado: {cache_meta_path}")

    images_dir = Path(args.images_dir)
    if not images_dir.exists():
        raise SystemExit(f"Pasta de imagens não existe: {images_dir}")

    output_dir = Path(args.output_dir) if args.output_dir else (run_dir / f"predict_unlabeled_{int(args.num_images)}")
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_meta = _load_json(cache_meta_path)
    model_id = str(cache_meta["model_id"])
    shortest_edge = int(cache_meta["shortest_edge"])
    crop_size = int(cache_meta["crop_size"])

    device = "cpu" if args.cpu else resolve_device()
    ckpt = torch.load(ckpt_path, map_location=device)
    label_to_idx = {str(k): int(v) for k, v in ckpt["label_to_idx"].items()}
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    in_dim = int(ckpt["in_dim"])
    dropout = float(ckpt.get("args", {}).get("dropout", 0.1))

    gt_stems: set[str] = set()
    if args.exclude_labels_dir:
        gt_stems |= _collect_gt_stems_from_labels_dir(Path(args.exclude_labels_dir))
    if args.exclude_images_dir:
        gt_stems |= _collect_gt_stems_from_images_dir(Path(args.exclude_images_dir))

    all_images = _collect_image_paths(images_dir)
    candidates = [p for p in all_images if p.stem not in gt_stems]
    if not candidates:
        raise SystemExit("Nenhuma imagem candidata após exclusão de GT.")

    rng = random.Random(int(args.seed))
    if int(args.num_images) > 0 and len(candidates) > int(args.num_images):
        selected = rng.sample(candidates, int(args.num_images))
    else:
        selected = candidates
    selected = sorted(selected)

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

    rows: list[dict[str, Any]] = []
    preprocess_ms: list[float] = []
    inference_ms: list[float] = []
    total_ms: list[float] = []

    t_all0 = time.perf_counter()
    with torch.no_grad():
        for p in selected:
            t0 = time.perf_counter()
            img = Image.open(p).convert("RGB")

            t_pre0 = time.perf_counter()
            proc_kwargs: dict[str, Any] = {}
            if shortest_edge > 0:
                proc_kwargs["size"] = {"shortest_edge": shortest_edge}
            if crop_size > 0:
                proc_kwargs["crop_size"] = {"height": crop_size, "width": crop_size}
            inputs = processor(images=img, return_tensors="pt", **proc_kwargs)
            pixel_values = inputs["pixel_values"].to(device)
            if device == "cuda":
                torch.cuda.synchronize()
            elif device == "mps":
                torch.mps.synchronize()
            t_pre1 = time.perf_counter()

            t_inf0 = time.perf_counter()
            feats = backbone._extract_features(pixel_values)
            logits = head(feats)
            probs = torch.softmax(logits, dim=1)
            conf, pred = torch.max(probs, dim=1)
            if device == "cuda":
                torch.cuda.synchronize()
            elif device == "mps":
                torch.mps.synchronize()
            t_inf1 = time.perf_counter()

            t1 = time.perf_counter()
            pre_ms = (t_pre1 - t_pre0) * 1000.0
            inf_ms = (t_inf1 - t_inf0) * 1000.0
            tot_ms = (t1 - t0) * 1000.0

            pred_idx = int(pred.item())
            rows.append(
                {
                    "image": str(p.resolve()),
                    "stem": p.stem,
                    "pred_idx": pred_idx,
                    "pred_label": idx_to_label[pred_idx],
                    "confidence": float(conf.item()),
                    "preprocess_ms": pre_ms,
                    "inference_ms": inf_ms,
                    "total_ms": tot_ms,
                }
            )
            preprocess_ms.append(pre_ms)
            inference_ms.append(inf_ms)
            total_ms.append(tot_ms)
    t_all1 = time.perf_counter()

    timing = {
        "wall_total_s": float(t_all1 - t_all0),
        "preprocess_mean_ms": _timing_stats(preprocess_ms)["mean_ms"],
        "preprocess_median_ms": _timing_stats(preprocess_ms)["median_ms"],
        "preprocess_p95_ms": _timing_stats(preprocess_ms)["p95_ms"],
        "inference_mean_ms": _timing_stats(inference_ms)["mean_ms"],
        "inference_median_ms": _timing_stats(inference_ms)["median_ms"],
        "inference_p95_ms": _timing_stats(inference_ms)["p95_ms"],
        "total_mean_ms": _timing_stats(total_ms)["mean_ms"],
        "total_median_ms": _timing_stats(total_ms)["median_ms"],
        "total_p95_ms": _timing_stats(total_ms)["p95_ms"],
    }

    pred_counts: dict[str, int] = {}
    for r in rows:
        pred_counts[r["pred_label"]] = pred_counts.get(r["pred_label"], 0) + 1
    pred_counts = dict(sorted(pred_counts.items(), key=lambda kv: kv[0]))

    summary = {
        "mode": "predict_unlabeled_grouped_html",
        "run_dir": str(run_dir.resolve()),
        "model_id": model_id,
        "device": device,
        "images_dir": str(images_dir.resolve()),
        "num_candidates_after_gt_filter": len(candidates),
        "num_images": len(rows),
        "num_images_requested": int(args.num_images),
        "num_gt_stems_excluded": len(gt_stems),
        "timing": timing,
        "pred_counts": pred_counts,
        "output_dir": str(output_dir.resolve()),
    }

    _save_json(output_dir / "summary_predict_unlabeled.json", summary)
    _save_json(output_dir / "predictions_rows.json", rows)
    (output_dir / "selected_images.txt").write_text(
        "\n".join(str(p.resolve()) for p in selected) + "\n",
        encoding="utf-8",
    )

    html_path = output_dir / "grouped_by_predicted_class.html"
    html_path.write_text(
        build_grouped_html(
            title=f"Predições Sem GT - Agrupado por Classe Predita ({len(rows)} imgs)",
            rows=rows,
            idx_to_label=idx_to_label,
            summary=summary,
        ),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"HTML: {html_path.resolve()}")


if __name__ == "__main__":
    main()
