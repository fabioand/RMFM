#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _pair_confusion(cm: np.ndarray, label_to_idx: dict[str, int], a: str, b: str) -> tuple[int, int, int]:
    ia = label_to_idx[a]
    ib = label_to_idx[b]
    return int(cm[ia, ib]), int(cm[ib, ia]), int(cm[ia, ib] + cm[ib, ia])


def _find_top_offdiag(cm: np.ndarray, idx_to_label: dict[int, str], top_k: int = 15) -> list[tuple[int, str, str]]:
    rows = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i == j:
                continue
            c = int(cm[i, j])
            if c > 0:
                rows.append((c, idx_to_label[i], idx_to_label[j]))
    rows.sort(reverse=True)
    return rows[:top_k]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compara dois runs do classificador periapical.")
    parser.add_argument("--baseline-run", required=True)
    parser.add_argument("--candidate-run", required=True)
    parser.add_argument("--output", default="", help="Default: <candidate-run>/compare_vs_baseline.json/.md")
    args = parser.parse_args()

    base_run = Path(args.baseline_run)
    cand_run = Path(args.candidate_run)

    base_summary = _load_json(base_run / "summary.json")
    cand_summary = _load_json(cand_run / "summary.json")
    base_report = _load_json(base_run / "classification_report_test.json")
    cand_report = _load_json(cand_run / "classification_report_test.json")
    base_label_to_idx = _load_json(base_run / "label_to_idx.json")
    cand_label_to_idx = _load_json(cand_run / "label_to_idx.json")

    if base_label_to_idx != cand_label_to_idx:
        raise SystemExit("label_to_idx difere entre os runs; comparação direta abortada.")

    label_to_idx = {str(k): int(v) for k, v in base_label_to_idx.items()}
    idx_to_label = {v: k for k, v in label_to_idx.items()}

    base_cm = np.loadtxt(base_run / "confusion_matrix_test.csv", delimiter=",").astype(int)
    cand_cm = np.loadtxt(cand_run / "confusion_matrix_test.csv", delimiter=",").astype(int)

    pairs_of_interest = [
        ("36-37-38", "48-47-46"),
        ("33", "43"),
        ("15-14", "24-25"),
        ("18-17-16", "26-27-28"),
    ]
    pair_rows = []
    for a, b in pairs_of_interest:
        if a in label_to_idx and b in label_to_idx:
            b_a2b, b_b2a, b_tot = _pair_confusion(base_cm, label_to_idx, a, b)
            c_a2b, c_b2a, c_tot = _pair_confusion(cand_cm, label_to_idx, a, b)
            pair_rows.append(
                {
                    "pair": f"{a} <-> {b}",
                    "baseline": {"a_to_b": b_a2b, "b_to_a": b_b2a, "total_swap": b_tot},
                    "candidate": {"a_to_b": c_a2b, "b_to_a": c_b2a, "total_swap": c_tot},
                    "delta_total_swap": int(c_tot - b_tot),
                }
            )

    per_class_delta = []
    for lbl in sorted(label_to_idx.keys()):
        f1_b = float(base_report.get(lbl, {}).get("f1-score", 0.0))
        f1_c = float(cand_report.get(lbl, {}).get("f1-score", 0.0))
        rec_b = float(base_report.get(lbl, {}).get("recall", 0.0))
        rec_c = float(cand_report.get(lbl, {}).get("recall", 0.0))
        per_class_delta.append(
            {
                "label": lbl,
                "delta_f1": f1_c - f1_b,
                "delta_recall": rec_c - rec_b,
                "baseline_f1": f1_b,
                "candidate_f1": f1_c,
            }
        )

    per_class_delta_sorted = sorted(per_class_delta, key=lambda x: x["delta_f1"], reverse=True)

    summary = {
        "baseline_run": str(base_run.resolve()),
        "candidate_run": str(cand_run.resolve()),
        "baseline": {
            "test_accuracy": float(base_summary["test"]["accuracy"]),
            "test_macro_f1": float(base_summary["test"]["macro_f1"]),
            "best_val_macro_f1": float(base_summary.get("best_val_macro_f1", 0.0)),
            "best_epoch": int(base_summary.get("best_epoch", -1)),
        },
        "candidate": {
            "test_accuracy": float(cand_summary["test"]["accuracy"]),
            "test_macro_f1": float(cand_summary["test"]["macro_f1"]),
            "best_val_macro_f1": float(cand_summary.get("best_val_macro_f1", 0.0)),
            "best_epoch": int(cand_summary.get("best_epoch", -1)),
        },
        "delta": {
            "test_accuracy": float(cand_summary["test"]["accuracy"] - base_summary["test"]["accuracy"]),
            "test_macro_f1": float(cand_summary["test"]["macro_f1"] - base_summary["test"]["macro_f1"]),
            "best_val_macro_f1": float(cand_summary.get("best_val_macro_f1", 0.0) - base_summary.get("best_val_macro_f1", 0.0)),
        },
        "pairs_of_interest": pair_rows,
        "top_gains_f1": per_class_delta_sorted[:8],
        "top_drops_f1": sorted(per_class_delta, key=lambda x: x["delta_f1"])[:8],
        "top_offdiag_baseline": _find_top_offdiag(base_cm, idx_to_label, top_k=12),
        "top_offdiag_candidate": _find_top_offdiag(cand_cm, idx_to_label, top_k=12),
    }

    out_base = Path(args.output) if args.output else (cand_run / "compare_vs_baseline")
    out_json = out_base.with_suffix(".json")
    out_md = out_base.with_suffix(".md")

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = []
    lines.append("# Comparação de Runs")
    lines.append("")
    lines.append(f"- Baseline: `{base_run}`")
    lines.append(f"- Candidato: `{cand_run}`")
    lines.append("")
    lines.append("## Métricas Globais")
    lines.append(f"- Test Accuracy: {summary['baseline']['test_accuracy']:.4f} -> {summary['candidate']['test_accuracy']:.4f} (Δ {summary['delta']['test_accuracy']:+.4f})")
    lines.append(f"- Test Macro F1: {summary['baseline']['test_macro_f1']:.4f} -> {summary['candidate']['test_macro_f1']:.4f} (Δ {summary['delta']['test_macro_f1']:+.4f})")
    lines.append(f"- Best Val Macro F1: {summary['baseline']['best_val_macro_f1']:.4f} -> {summary['candidate']['best_val_macro_f1']:.4f} (Δ {summary['delta']['best_val_macro_f1']:+.4f})")
    lines.append("")
    lines.append("## Pares Espelhados de Interesse")
    lines.append("| Par | Baseline swaps | Candidato swaps | Δ swaps |")
    lines.append("|---|---:|---:|---:|")
    for row in pair_rows:
        lines.append(
            f"| {row['pair']} | {row['baseline']['total_swap']} | {row['candidate']['total_swap']} | {row['delta_total_swap']:+d} |"
        )
    lines.append("")
    lines.append("## Maiores Ganhos de F1 por Classe")
    lines.append("| Classe | F1 baseline | F1 candidato | Δ F1 |")
    lines.append("|---|---:|---:|---:|")
    for row in summary["top_gains_f1"]:
        lines.append(f"| {row['label']} | {row['baseline_f1']:.3f} | {row['candidate_f1']:.3f} | {row['delta_f1']:+.3f} |")
    lines.append("")
    lines.append("## Maiores Quedas de F1 por Classe")
    lines.append("| Classe | F1 baseline | F1 candidato | Δ F1 |")
    lines.append("|---|---:|---:|---:|")
    for row in summary["top_drops_f1"]:
        lines.append(f"| {row['label']} | {row['baseline_f1']:.3f} | {row['candidate_f1']:.3f} | {row['delta_f1']:+.3f} |")
    lines.append("")
    lines.append(f"JSON detalhado: `{out_json}`")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(summary["delta"], ensure_ascii=False, indent=2))
    print(f"JSON: {out_json}")
    print(f"MD: {out_md}")


if __name__ == "__main__":
    main()
