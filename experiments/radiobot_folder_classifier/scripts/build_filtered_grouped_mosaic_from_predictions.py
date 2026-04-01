#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fnmatch
import html
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _load_prediction_jsons(predictions_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for p in sorted(predictions_dir.glob("*.json")):
        if p.name == "_summary.json":
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        pred_label = str(data.get("pred_label", "")).strip()
        image_path = str(data.get("image_path", "")).strip()
        if not pred_label or not image_path:
            continue
        data["_json_path"] = str(p.resolve())
        rows.append(data)
    return rows


def _label_excluded(label: str, patterns: list[str]) -> bool:
    low = label.lower()
    for pat in patterns:
        if fnmatch.fnmatch(low, pat.lower()):
            return True
    return False


def _sanitize_label_for_path(label: str) -> str:
    out = re.sub(r"[^a-zA-Z0-9._-]+", "_", label.strip())
    return out.strip("_") or "class"


def _build_html(rows_by_class: dict[str, list[dict[str, Any]]], summary: dict[str, Any]) -> str:
    sections: list[str] = []
    for label, items in rows_by_class.items():
        cards: list[str] = []
        for r in items:
            img_src = html.escape(str(r["html_image_src"]))
            stem = html.escape(str(r.get("stem", "")))
            pred_label = html.escape(str(r.get("pred_label", "")))
            conf = float(r.get("pred_confidence", 0.0))
            image_path = html.escape(str(r.get("image_path", "")))
            cards.append(
                f"""
                <article class="card">
                  <div class="thumb"><img src="{img_src}" alt="{stem}" loading="lazy" /></div>
                  <div class="body">
                    <h3 title="{image_path}">{stem}</h3>
                    <p><b>Pred:</b> {pred_label}</p>
                    <p><b>Conf:</b> {conf:.3f}</p>
                  </div>
                </article>
                """
            )
        sections.append(
            f"""
            <section class="cluster">
              <h2>{html.escape(label)} - {len(items)} imgs</h2>
              <div class="grid">{''.join(cards)}</div>
            </section>
            """
        )

    excluded = ", ".join(summary["exclude_patterns"]) if summary["exclude_patterns"] else "(nenhum)"
    return f"""<!doctype html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Mosaico de Inspeção - Predições</title>
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
    .wrap {{ max-width: 1600px; margin: 0 auto; padding: 24px; }}
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
    <h1>Mosaico de Inspeção por Classe Predita</h1>
    <p class="meta">
      total_json={summary['num_input_json']} | total_filtrado={summary['num_selected_rows']} |
      classes={summary['num_classes_selected']} | ordenação=menos->mais | excluídas={html.escape(excluded)}
    </p>
    {''.join(sections)}
  </main>
</body>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gera mosaico HTML agrupado por classe predita a partir de JSONs de predição."
    )
    parser.add_argument("--predictions-dir", required=True, help="Pasta com JSONs por imagem (inclui _summary.json).")
    parser.add_argument("--output-dir", required=True, help="Pasta de saída do mosaico.")
    parser.add_argument(
        "--exclude-label-pattern",
        action="append",
        default=[],
        help="Pattern fnmatch para excluir classes (pode repetir). Ex.: 'Periapical', 'Fotografia*'.",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=0,
        help="Limita amostras por classe no HTML (0 = sem limite).",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1000,
        help="Imprime progresso a cada N imagens processadas para o mosaico.",
    )
    args = parser.parse_args()
    t0 = time.perf_counter()

    predictions_dir = Path(args.predictions_dir).resolve()
    if not predictions_dir.exists():
        raise SystemExit(f"predictions-dir não existe: {predictions_dir}")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    images_out_dir = output_dir / "images"
    images_out_dir.mkdir(parents=True, exist_ok=True)

    default_excludes = ["Periapical", "Fotografia*", "Intra-Oral*"]
    exclude_patterns = default_excludes + list(args.exclude_label_pattern)

    print("[1/5] Carregando JSONs de predição...")
    all_rows = _load_prediction_jsons(predictions_dir)
    if not all_rows:
        raise SystemExit("Nenhum JSON de predição válido encontrado.")
    print(f"       jsons válidos: {len(all_rows)}")

    print("[2/5] Aplicando filtros de classe...")
    selected = [r for r in all_rows if not _label_excluded(str(r.get("pred_label", "")), exclude_patterns)]
    if not selected:
        raise SystemExit("Nenhuma linha restante após aplicar filtros de classe.")
    print(f"       após filtro: {len(selected)}")

    grouped_raw: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in selected:
        grouped_raw[str(r["pred_label"])].append(r)

    ordered_labels = sorted(grouped_raw.keys(), key=lambda c: (len(grouped_raw[c]), c.lower()))
    print(f"[3/5] Montando grupos ordenados (menos -> mais): {len(ordered_labels)} classes")

    rows_by_class: dict[str, list[dict[str, Any]]] = {}
    class_counts_selected: dict[str, int] = {}
    total_symlink_fail = 0
    idx_global = 0

    progress_every = max(1, int(args.progress_every))
    next_mark = progress_every
    total_target = sum(len(grouped_raw[k]) if int(args.max_per_class) <= 0 else min(len(grouped_raw[k]), int(args.max_per_class)) for k in ordered_labels)

    print("[4/5] Criando symlinks e preparando cards...")
    for label in ordered_labels:
        items = sorted(grouped_raw[label], key=lambda x: float(x.get("pred_confidence", 0.0)), reverse=True)
        if int(args.max_per_class) > 0:
            items = items[: int(args.max_per_class)]
        class_counts_selected[label] = len(items)

        safe_label = _sanitize_label_for_path(label)
        class_img_dir = images_out_dir / safe_label
        class_img_dir.mkdir(parents=True, exist_ok=True)

        out_items: list[dict[str, Any]] = []
        for r in items:
            idx_global += 1
            src = Path(str(r["image_path"]))
            link_name = f"{idx_global:07d}_{src.name}"
            link_path = class_img_dir / link_name
            try:
                if link_path.exists() or link_path.is_symlink():
                    link_path.unlink()
                link_path.symlink_to(src.resolve())
                r["html_image_src"] = str(Path("images") / safe_label / link_name)
                out_items.append(r)
            except Exception:
                total_symlink_fail += 1
                continue
            if idx_global >= next_mark:
                pct = (100.0 * idx_global) / max(1, total_target)
                print(f"       progresso: {idx_global}/{total_target} ({pct:.1f}%)")
                while idx_global >= next_mark:
                    next_mark += progress_every
        rows_by_class[label] = out_items

    print("[5/5] Gravando HTML + resumos...")
    summary = {
        "mode": "build_filtered_grouped_mosaic_from_predictions",
        "predictions_dir": str(predictions_dir),
        "output_dir": str(output_dir),
        "exclude_patterns": exclude_patterns,
        "num_input_json": len(all_rows),
        "num_selected_rows": sum(len(v) for v in rows_by_class.values()),
        "num_classes_selected": len(rows_by_class),
        "class_counts_selected": class_counts_selected,
        "max_per_class": int(args.max_per_class),
        "num_symlink_fail": int(total_symlink_fail),
    }

    (output_dir / "summary_filtered_grouped.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "rows_filtered_grouped.json").write_text(
        json.dumps(rows_by_class, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    html_text = _build_html(rows_by_class=rows_by_class, summary=summary)
    html_path = output_dir / "grouped_filtered_by_class.html"
    html_path.write_text(html_text, encoding="utf-8")
    t1 = time.perf_counter()
    print(f"       tempo_total_s: {t1 - t0:.2f}")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"HTML: {html_path}")


if __name__ == "__main__":
    main()
