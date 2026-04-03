#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    matplotlib = None
    plt = None

from PIL import Image, ImageDraw, ImageFont


@dataclass
class ErrorRow:
    true_idx: int
    pred_idx: int
    true_label: str
    pred_label: str
    count: int
    true_total: int
    error_rate_within_true: float
    error_share_global: float


def _load_label_order(label_to_idx_path: Path) -> list[str]:
    payload = json.loads(label_to_idx_path.read_text(encoding="utf-8"))
    idx_to_label = {int(v): str(k) for k, v in payload.items()}
    n = max(idx_to_label.keys()) + 1 if idx_to_label else 0
    return [idx_to_label[i] for i in range(n)]


def _load_confusion(path: Path) -> np.ndarray:
    cm = np.loadtxt(path, delimiter=",", dtype=np.int64)
    if cm.ndim == 1:
        cm = cm.reshape(1, -1)
    return cm


def _extract_errors(cm: np.ndarray, labels: list[str]) -> tuple[list[ErrorRow], int]:
    if cm.shape[0] != cm.shape[1]:
        raise ValueError(f"Matriz de confusão não quadrada: {cm.shape}")
    if cm.shape[0] != len(labels):
        raise ValueError(
            f"Número de labels ({len(labels)}) difere da matriz ({cm.shape[0]})."
        )

    true_totals = cm.sum(axis=1)
    total_errors = int(cm.sum() - np.trace(cm))
    rows: list[ErrorRow] = []

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i == j:
                continue
            c = int(cm[i, j])
            if c <= 0:
                continue
            true_total = int(true_totals[i])
            rows.append(
                ErrorRow(
                    true_idx=i,
                    pred_idx=j,
                    true_label=labels[i],
                    pred_label=labels[j],
                    count=c,
                    true_total=true_total,
                    error_rate_within_true=(float(c) / max(1, true_total)),
                    error_share_global=(float(c) / max(1, total_errors)),
                )
            )

    rows.sort(key=lambda r: r.count, reverse=True)
    return rows, total_errors


def _save_csv(path: Path, rows: list[ErrorRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "true_idx",
                "pred_idx",
                "true_label",
                "pred_label",
                "count",
                "true_total",
                "error_rate_within_true",
                "error_share_global",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.true_idx,
                    r.pred_idx,
                    r.true_label,
                    r.pred_label,
                    r.count,
                    r.true_total,
                    f"{r.error_rate_within_true:.8f}",
                    f"{r.error_share_global:.8f}",
                ]
            )


def _plot_histogram_matplotlib(path: Path, rows: list[ErrorRow], title: str, top_k: int) -> None:
    top = rows[:top_k]
    if not top:
        fig, ax = plt.subplots(figsize=(8, 3), dpi=160)
        ax.text(0.5, 0.5, "Sem erros para plotar.", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return

    labels = [f"{r.true_label} -> {r.pred_label}" for r in top]
    counts = [r.count for r in top]

    # Plot invertido para o maior aparecer no topo visualmente.
    labels = labels[::-1]
    counts = counts[::-1]

    h = max(4.0, 0.35 * len(labels) + 1.5)
    fig, ax = plt.subplots(figsize=(12, h), dpi=160)
    ax.barh(labels, counts, color="#D65F5F")
    ax.set_title(title, fontsize=12, weight="bold")
    ax.set_xlabel("Contagem de Erros")
    ax.set_ylabel("Classe Real -> Classe Predita")
    for i, c in enumerate(counts):
        ax.text(c + 0.1, i, str(c), va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _plot_histogram_pillow(path: Path, rows: list[ErrorRow], title: str, top_k: int) -> None:
    top = rows[:top_k]
    labels = [f"{r.true_label} -> {r.pred_label}" for r in top]
    counts = [r.count for r in top]

    if not top:
        img = Image.new("RGB", (900, 300), "white")
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()
        draw.text((30, 30), title, fill="black", font=font)
        draw.text((30, 80), "Sem erros para plotar.", fill="black", font=font)
        img.save(path)
        return

    width = 1400
    row_h = 28
    margin_top = 70
    margin_left = 420
    margin_right = 80
    margin_bottom = 40
    height = margin_top + margin_bottom + row_h * len(labels)

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    draw.text((20, 20), title, fill="black", font=font)
    draw.text((20, 40), "Classe real -> classe predita", fill="#444444", font=font)

    max_count = max(counts)
    bar_max_w = width - margin_left - margin_right

    for i, (lab, c) in enumerate(zip(labels, counts)):
        y = margin_top + i * row_h
        draw.text((10, y + 6), lab[:65], fill="black", font=font)
        bw = int((c / max_count) * bar_max_w) if max_count > 0 else 0
        draw.rectangle([margin_left, y + 4, margin_left + bw, y + row_h - 6], fill="#D65F5F")
        draw.text((margin_left + bw + 8, y + 6), str(c), fill="black", font=font)

    img.save(path)


def _plot_histogram(path: Path, rows: list[ErrorRow], title: str, top_k: int) -> str:
    if plt is not None:
        _plot_histogram_matplotlib(path, rows, title, top_k)
        return "matplotlib"
    _plot_histogram_pillow(path, rows, title, top_k)
    return "pillow"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gera histograma dos tipos de erro (classe real X para classe predita Y) a partir da matriz de confusão."
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Diretório da run com confusion_matrix_test.csv e label_to_idx.json (ex.: .../e2_retrain_head).",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Diretório de saída. Default: <run-dir>/error_analysis",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=30,
        help="Número de tipos de erro exibidos no histograma.",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=1,
        help="Filtra tipos de erro com contagem menor que este valor.",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else (run_dir / "error_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    cm_path = run_dir / "confusion_matrix_test.csv"
    labels_path = run_dir / "label_to_idx.json"
    if not cm_path.exists():
        raise SystemExit(f"Matriz de confusão não encontrada: {cm_path}")
    if not labels_path.exists():
        raise SystemExit(f"label_to_idx.json não encontrado: {labels_path}")

    labels = _load_label_order(labels_path)
    cm = _load_confusion(cm_path)
    rows, total_errors = _extract_errors(cm, labels)
    rows = [r for r in rows if r.count >= int(args.min_count)]

    full_csv = output_dir / "error_types_full.csv"
    top_csv = output_dir / "error_types_top.csv"
    hist_png = output_dir / "error_types_histogram_top.png"
    summary_json = output_dir / "error_types_summary.json"

    _save_csv(full_csv, rows)
    _save_csv(top_csv, rows[: int(args.top_k)])
    renderer = _plot_histogram(
        path=hist_png,
        rows=rows,
        top_k=int(args.top_k),
        title=f"Top {int(args.top_k)} tipos de erro - classificação periapical",
    )

    summary = {
        "run_dir": str(run_dir),
        "num_classes": int(cm.shape[0]),
        "num_samples_test": int(cm.sum()),
        "num_total_errors": int(total_errors),
        "num_error_types": int(len(rows)),
        "top_k": int(args.top_k),
        "min_count": int(args.min_count),
        "histogram_renderer": renderer,
        "paths": {
            "full_csv": str(full_csv),
            "top_csv": str(top_csv),
            "histogram_png": str(hist_png),
        },
        "top5": [
            {
                "true_label": r.true_label,
                "pred_label": r.pred_label,
                "count": int(r.count),
                "error_rate_within_true": float(r.error_rate_within_true),
                "error_share_global": float(r.error_share_global),
            }
            for r in rows[:5]
        ],
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
