#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor

from dino_periapical_cls.data import (
    PeriapicalDataset,
    build_label_mapping,
    discover_samples,
    make_collate_fn,
    stratified_split,
)
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


def build_grouped_html(title: str, rows: list[dict[str, Any]], idx_to_label: dict[int, str]) -> str:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        grouped[int(r["pred_idx"])].append(r)

    sections = []
    for pred_idx in sorted(grouped.keys()):
        items = grouped[pred_idx]
        cls_name = idx_to_label[pred_idx]
        correct = sum(1 for x in items if x["true_idx"] == x["pred_idx"])
        cards = []
        for item in sorted(items, key=lambda x: x["confidence"], reverse=True):
            img_uri = Path(item["image"]).resolve().as_uri()
            ok = item["true_idx"] == item["pred_idx"]
            cards.append(
                f"""
                <article class="card {'ok' if ok else 'err'}">
                  <div class="thumb"><img src="{img_uri}" alt="{html.escape(item['stem'])}" loading="lazy" /></div>
                  <div class="body">
                    <h3 title="{html.escape(item['image'])}">{html.escape(item['stem'])}</h3>
                    <p><b>Pred:</b> {html.escape(item['pred_label'])}</p>
                    <p><b>True:</b> {html.escape(item['true_label'])}</p>
                    <p><b>Conf:</b> {item['confidence']:.3f}</p>
                    <p><b>Status:</b> {'OK' if ok else 'ERRO'}</p>
                  </div>
                </article>
                """
            )

        sections.append(
            f"""
            <section class="cluster">
              <h2>{html.escape(cls_name)} (pred) - {len(items)} imgs | acertos: {correct}</h2>
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
      --ok: #15803d;
      --err: #b91c1c;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: var(--ink); background: var(--bg); }}
    .wrap {{ max-width: 1540px; margin: 0 auto; padding: 24px; }}
    h1 {{ margin: 0 0 14px; font-size: 24px; }}
    .cluster {{ margin-top: 18px; }}
    h2 {{ margin: 0 0 10px; font-size: 17px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(230px, 1fr)); gap: 12px; }}
    .card {{ border: 1px solid var(--line); border-radius: 12px; background: var(--card); overflow: hidden; }}
    .card.ok {{ border-color: #bbf7d0; }}
    .card.err {{ border-color: #fecaca; }}
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
    {''.join(sections)}
  </main>
</body>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Avalia checkpoint no split test e gera HTML agrupado por classe predita.")
    parser.add_argument("--run-dir", required=True, help="Pasta do run (onde estão best_head_only.pt e features_cache/cache_meta.json)")
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--labels-dir", required=True)
    parser.add_argument("--output-dir", default="", help="Default: <run-dir>/eval_test_grouped")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    ckpt_path = run_dir / "best_head_only.pt"
    cache_meta_path = run_dir / "features_cache" / "cache_meta.json"
    if not ckpt_path.exists():
        raise SystemExit(f"Checkpoint não encontrado: {ckpt_path}")
    if not cache_meta_path.exists():
        raise SystemExit(f"cache_meta.json não encontrado: {cache_meta_path}")

    output_dir = Path(args.output_dir) if args.output_dir else (run_dir / "eval_test_grouped")
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_meta = _load_json(cache_meta_path)
    seed = int(cache_meta["seed"])
    val_size = float(cache_meta["val_size"])
    test_size = float(cache_meta["test_size"])
    max_samples = int(cache_meta["max_samples"])
    model_id = str(cache_meta["model_id"])
    shortest_edge = int(cache_meta["shortest_edge"])
    crop_size = int(cache_meta["crop_size"])

    device = "cpu" if args.cpu else resolve_device()
    ckpt = torch.load(ckpt_path, map_location=device)
    label_to_idx_ckpt = {str(k): int(v) for k, v in ckpt["label_to_idx"].items()}
    idx_to_label = {v: k for k, v in label_to_idx_ckpt.items()}
    in_dim = int(ckpt["in_dim"])
    dropout = float(ckpt.get("args", {}).get("dropout", 0.1))

    samples = discover_samples(Path(args.images_dir), Path(args.labels_dir))
    if max_samples > 0:
        samples = samples[:max_samples]
    label_to_idx_now = build_label_mapping(samples)
    if label_to_idx_now != label_to_idx_ckpt:
        raise SystemExit("Mapping de classes atual difere do mapping do checkpoint. Abortei para evitar avaliação inconsistente.")

    splits = stratified_split(samples, val_size=val_size, test_size=test_size, seed=seed)
    ds_test = PeriapicalDataset(splits["test"], label_to_idx_ckpt)

    processor = AutoImageProcessor.from_pretrained(
        model_id,
        local_files_only=bool(args.offline),
    )
    collate_fn = make_collate_fn(processor, shortest_edge=shortest_edge, crop_size=crop_size)
    test_loader = DataLoader(
        ds_test,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        collate_fn=collate_fn,
        pin_memory=False,
    )

    backbone = FrozenDinoClassifier(
        model_id=model_id,
        num_classes=len(label_to_idx_ckpt),
        local_files_only=bool(args.offline),
        freeze_backbone=True,
    ).to(device)
    backbone.eval()

    head = HeadClassifier(in_dim=in_dim, num_classes=len(label_to_idx_ckpt), dropout=dropout).to(device)
    head.load_state_dict(ckpt["head_state_dict"])
    head.eval()

    y_true: list[int] = []
    y_pred: list[int] = []
    losses: list[float] = []
    rows: list[dict[str, Any]] = []

    with torch.no_grad():
        for batch in test_loader:
            pixel_values = batch["pixel_values"].to(device)
            labels = torch.tensor(batch["labels"], dtype=torch.long, device=device)
            feats = backbone._extract_features(pixel_values)
            logits = head(feats)
            probs = torch.softmax(logits, dim=1)
            conf, pred = torch.max(probs, dim=1)
            loss = F.cross_entropy(logits, labels)

            losses.append(float(loss.item()))
            y_true.extend(labels.detach().cpu().tolist())
            y_pred.extend(pred.detach().cpu().tolist())

            for i in range(len(batch["paths"])):
                true_idx = int(labels[i].item())
                pred_idx = int(pred[i].item())
                rows.append(
                    {
                        "image": batch["paths"][i],
                        "stem": batch["stems"][i],
                        "true_idx": true_idx,
                        "pred_idx": pred_idx,
                        "true_label": idx_to_label[true_idx],
                        "pred_label": idx_to_label[pred_idx],
                        "confidence": float(conf[i].item()),
                    }
                )

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(idx_to_label))),
        target_names=[idx_to_label[i] for i in range(len(idx_to_label))],
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(idx_to_label))))

    summary = {
        "run_dir": str(run_dir.resolve()),
        "mode": "eval_test_grouped_html",
        "model_id": model_id,
        "device": device,
        "num_test_samples": len(y_true),
        "test_loss": float(np.mean(losses)) if losses else 0.0,
        "test_accuracy": float(accuracy_score(y_true, y_pred)),
        "test_macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "output_dir": str(output_dir.resolve()),
    }

    _save_json(output_dir / "summary_eval.json", summary)
    _save_json(output_dir / "classification_report_test_eval.json", report)
    _save_json(output_dir / "predictions_test_rows.json", rows)
    np.savetxt(output_dir / "confusion_matrix_test_eval.csv", cm, fmt="%d", delimiter=",")

    html_path = output_dir / "grouped_by_predicted_class.html"
    html_path.write_text(
        build_grouped_html(
            title=f"Classificação Test - Agrupado por Classe Predita ({len(rows)} imgs)",
            rows=rows,
            idx_to_label=idx_to_label,
        ),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"HTML: {html_path.resolve()}")


if __name__ == "__main__":
    main()

