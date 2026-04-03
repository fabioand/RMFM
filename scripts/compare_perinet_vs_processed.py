#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class Item:
    key: str
    path: Path
    cls: str


def _is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in VALID_EXT


def _key_for(path: Path, match_by: str) -> str:
    if match_by == "name":
        return path.name
    return path.stem


def _scan_perinet(root: Path, match_by: str) -> list[Item]:
    items: list[Item] = []
    # Convenção esperada: root/{1..14}/<arquivos>
    for cls_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        cls = cls_dir.name
        for p in cls_dir.rglob("*"):
            if not _is_image(p):
                continue
            items.append(Item(key=_key_for(p, match_by), path=p.resolve(), cls=cls))
    return items


def _scan_processed(imgs_dir: Path, match_by: str) -> list[Item]:
    items: list[Item] = []
    for p in imgs_dir.rglob("*"):
        if not _is_image(p):
            continue
        items.append(Item(key=_key_for(p, match_by), path=p.resolve(), cls="processed"))
    return items


def _write_txt(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compara imagens do Perinet (classes 1..14) com /periapicais_processed/imgs e reporta interseção."
    )
    parser.add_argument("--perinet-root", required=True, help="Ex.: /dataminer/Peris/Perinet")
    parser.add_argument(
        "--processed-imgs-dir",
        required=True,
        help="Ex.: /dataminer/rmdatasets/data/periapicais_processed/imgs",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Diretório para salvar relatórios (json/csv/txt).",
    )
    parser.add_argument(
        "--match-by",
        choices=("stem", "name"),
        default="stem",
        help="Critério de matching: stem (default) ou nome completo com extensão.",
    )
    args = parser.parse_args()

    perinet_root = Path(args.perinet_root).resolve()
    processed_dir = Path(args.processed_imgs_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not perinet_root.exists():
        raise SystemExit(f"--perinet-root não existe: {perinet_root}")
    if not processed_dir.exists():
        raise SystemExit(f"--processed-imgs-dir não existe: {processed_dir}")

    perinet_items = _scan_perinet(perinet_root, args.match_by)
    processed_items = _scan_processed(processed_dir, args.match_by)

    perinet_by_key: dict[str, list[Item]] = defaultdict(list)
    for it in perinet_items:
        perinet_by_key[it.key].append(it)

    processed_by_key: dict[str, list[Item]] = defaultdict(list)
    for it in processed_items:
        processed_by_key[it.key].append(it)

    perinet_keys = set(perinet_by_key.keys())
    processed_keys = set(processed_by_key.keys())

    common_keys = sorted(perinet_keys & processed_keys)
    only_perinet_keys = sorted(perinet_keys - processed_keys)
    only_processed_keys = sorted(processed_keys - perinet_keys)

    common_rows: list[dict[str, str]] = []
    common_by_class_key: dict[str, set[str]] = defaultdict(set)
    for k in common_keys:
        for pit in perinet_by_key[k]:
            for qit in processed_by_key[k]:
                common_rows.append(
                    {
                        "key": k,
                        "perinet_class": pit.cls,
                        "perinet_path": str(pit.path),
                        "processed_path": str(qit.path),
                    }
                )
            common_by_class_key[pit.cls].add(k)

    class_report = []
    for cls in sorted({it.cls for it in perinet_items}):
        per_cls_keys = {it.key for it in perinet_items if it.cls == cls}
        common_cls_keys = common_by_class_key.get(cls, set())
        only_cls_keys = per_cls_keys - processed_keys
        class_report.append(
            {
                "class": cls,
                "perinet_unique_keys": len(per_cls_keys),
                "common_keys": len(common_cls_keys),
                "only_perinet_keys": len(only_cls_keys),
                "coverage_common_over_perinet": (
                    float(len(common_cls_keys)) / max(1, len(per_cls_keys))
                ),
            }
        )

    duplicates = {
        "perinet_duplicate_keys": int(sum(1 for _, v in perinet_by_key.items() if len(v) > 1)),
        "processed_duplicate_keys": int(sum(1 for _, v in processed_by_key.items() if len(v) > 1)),
    }

    summary = {
        "perinet_root": str(perinet_root),
        "processed_imgs_dir": str(processed_dir),
        "match_by": args.match_by,
        "counts": {
            "perinet_files": len(perinet_items),
            "processed_files": len(processed_items),
            "perinet_unique_keys": len(perinet_keys),
            "processed_unique_keys": len(processed_keys),
            "common_unique_keys": len(common_keys),
            "only_perinet_unique_keys": len(only_perinet_keys),
            "only_processed_unique_keys": len(only_processed_keys),
        },
        "duplicates": duplicates,
        "class_report": class_report,
        "paths": {
            "summary_json": str((out_dir / "summary.json").resolve()),
            "common_keys_txt": str((out_dir / "common_keys.txt").resolve()),
            "only_perinet_keys_txt": str((out_dir / "only_perinet_keys.txt").resolve()),
            "only_processed_keys_txt": str((out_dir / "only_processed_keys.txt").resolve()),
            "common_pairs_csv": str((out_dir / "common_pairs.csv").resolve()),
            "class_report_csv": str((out_dir / "class_report.csv").resolve()),
        },
    }

    # Saídas
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_txt(out_dir / "common_keys.txt", common_keys)
    _write_txt(out_dir / "only_perinet_keys.txt", only_perinet_keys)
    _write_txt(out_dir / "only_processed_keys.txt", only_processed_keys)
    _write_csv(
        out_dir / "common_pairs.csv",
        common_rows,
        fieldnames=["key", "perinet_class", "perinet_path", "processed_path"],
    )
    _write_csv(
        out_dir / "class_report.csv",
        [
            {
                "class": str(r["class"]),
                "perinet_unique_keys": str(r["perinet_unique_keys"]),
                "common_keys": str(r["common_keys"]),
                "only_perinet_keys": str(r["only_perinet_keys"]),
                "coverage_common_over_perinet": f'{r["coverage_common_over_perinet"]:.8f}',
            }
            for r in class_report
        ],
        fieldnames=[
            "class",
            "perinet_unique_keys",
            "common_keys",
            "only_perinet_keys",
            "coverage_common_over_perinet",
        ],
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

