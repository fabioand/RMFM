#!/usr/bin/env python3
"""Analyze per-tooth anomalies from data_anomalie_laudo JSON files.

Expected JSON shape per file:
{
  "16": ["imagem radiolucida na coroa"],
  "13": ["calculo salivar", ...]
}
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable


def normalize_text(value: str) -> str:
    """Normalize anomaly labels for stable counting."""
    text = value.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def iter_json_files(input_dir: Path) -> Iterable[Path]:
    return sorted(p for p in input_dir.glob("*.json") if p.is_file())


def extract_labels(raw_value: object) -> list[str]:
    if raw_value is None:
        return []
    if isinstance(raw_value, list):
        values = raw_value
    else:
        values = [raw_value]
    labels: list[str] = []
    for item in values:
        text = str(item).strip()
        if text:
            labels.append(normalize_text(text))
    return labels


def ascii_histogram(counter: Counter[str], top_n: int | None = None, width: int = 40) -> str:
    items = counter.most_common(top_n)
    if not items:
        return "(vazio)"
    max_count = items[0][1]
    lines: list[str] = []
    for label, count in items:
        bar_len = 1 if max_count == 0 else max(1, round((count / max_count) * width))
        lines.append(f"{label:>55} | {'#' * bar_len} {count}")
    return "\n".join(lines)


def write_counter_csv(path: Path, header_name: str, counter: Counter[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([header_name, "count"])
        for label, count in counter.most_common():
            writer.writerow([label, count])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Varre JSONs da pasta data_anomalie_laudo e gera contagem de anomalias "
            "e contagem de dentes com presenca de anomalias."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/Users/fabioandrade/RMFM/periapicais_processed_sample/data_anomalie_laudo"),
        help="Pasta com JSONs do tipo data_anomalie_laudo.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/Users/fabioandrade/RMFM/out/anomalie_laudo_analysis"),
        help="Pasta de saida para CSVs e resumo.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=30,
        help="Quantidade maxima de linhas no histograma ASCII impresso no terminal.",
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    anomaly_counts: Counter[str] = Counter()
    tooth_presence_counts: Counter[str] = Counter()
    invalid_files: list[dict[str, str]] = []

    files = list(iter_json_files(input_dir))
    for path in files:
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:  # noqa: BLE001
            invalid_files.append({"file": str(path), "error": str(exc)})
            continue

        if not isinstance(data, dict):
            invalid_files.append({"file": str(path), "error": "root_not_dict"})
            continue

        for tooth, raw_value in data.items():
            labels = extract_labels(raw_value)
            if labels:
                tooth_presence_counts[str(tooth)] += 1
                for label in labels:
                    anomaly_counts[label] += 1

    write_counter_csv(output_dir / "anomaly_histogram.csv", "anomaly", anomaly_counts)
    write_counter_csv(output_dir / "tooth_presence_histogram.csv", "tooth", tooth_presence_counts)

    summary = {
        "input_dir": str(input_dir),
        "total_json_files": len(files),
        "valid_json_files": len(files) - len(invalid_files),
        "invalid_json_files": len(invalid_files),
        "unique_anomalies": len(anomaly_counts),
        "unique_teeth_with_presence": len(tooth_presence_counts),
        "top_anomalies": anomaly_counts.most_common(20),
        "top_teeth_with_presence": tooth_presence_counts.most_common(20),
        "invalid_files": invalid_files,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with (output_dir / "unique_anomalies.txt").open("w", encoding="utf-8") as f:
        for label, count in anomaly_counts.most_common():
            f.write(f"{label}\t{count}\n")

    print("=== ANALISE data_anomalie_laudo ===")
    print(f"Entrada: {input_dir}")
    print(f"Saida:   {output_dir}")
    print(f"Arquivos JSON encontrados: {len(files)}")
    print(f"Arquivos invalidos:        {len(invalid_files)}")
    print(f"Anomalias unicas:          {len(anomaly_counts)}")
    print(f"Dentes com presenca:       {len(tooth_presence_counts)}")
    print("")
    print("Histograma - anomalias (top):")
    print(ascii_histogram(anomaly_counts, top_n=args.top))
    print("")
    print("Histograma - dentes com presenca (top):")
    print(ascii_histogram(tooth_presence_counts, top_n=args.top))


if __name__ == "__main__":
    main()
