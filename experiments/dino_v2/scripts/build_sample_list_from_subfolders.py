#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _is_junk_path(root: Path, p: Path) -> bool:
    rel = p.relative_to(root)
    if any(part == "__MACOSX" for part in rel.parts):
        return True
    # AppleDouble sidecar files from zip on macOS
    if p.name.startswith("._"):
        return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Amostra N imagens por pasta interna e salva um JSON de lista para uso em embeddings/cluster."
    )
    parser.add_argument("--root-dir", required=True, help="Pasta raiz que contem subpastas com imagens.")
    parser.add_argument("--n-per-folder", type=int, required=True, help="Numero de amostras por pasta.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-json",
        default="",
        help="Caminho do JSON de saida. Default: <root-dir>/sample_list_n<N>.json",
    )
    parser.add_argument(
        "--include-root-images",
        action="store_true",
        help="Inclui imagens diretamente na raiz (grupo '.').",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Percorre subpastas recursivamente (default: apenas 1 nivel).",
    )
    args = parser.parse_args()

    root = Path(args.root_dir).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"Pasta raiz invalida: {root}")
    if args.n_per_folder <= 0:
        raise SystemExit("--n-per-folder deve ser > 0")

    groups: dict[str, list[Path]] = defaultdict(list)
    if args.recursive:
        candidates = sorted(root.rglob("*"))
    else:
        candidates = []
        if args.include_root_images:
            candidates.extend(sorted(root.iterdir()))
        for child in sorted(root.iterdir()):
            if child.is_dir():
                candidates.extend(sorted(child.iterdir()))

    for p in candidates:
        if not p.is_file() or p.suffix.lower() not in VALID_EXT:
            continue
        if _is_junk_path(root, p):
            continue
        rel_parent = p.parent.relative_to(root)
        if not args.include_root_images and str(rel_parent) == ".":
            continue
        folder_key = str(rel_parent)
        groups[folder_key].append(p)

    if not groups:
        raise SystemExit("Nenhuma imagem encontrada nas subpastas.")

    rng = random.Random(int(args.seed))
    sampled_items: list[dict[str, str]] = []
    folders_meta: list[dict[str, int | str]] = []

    for folder_key in sorted(groups.keys()):
        imgs = groups[folder_key]
        k = min(len(imgs), int(args.n_per_folder))
        selected = rng.sample(imgs, k) if len(imgs) > k else imgs
        selected = sorted(selected)
        for p in selected:
            sampled_items.append(
                {
                    "folder": folder_key,
                    "path": str(p.relative_to(root)),
                }
            )
        folders_meta.append(
            {
                "folder": folder_key,
                "num_available": len(imgs),
                "num_selected": len(selected),
            }
        )

    out = {
        "root_dir": str(root),
        "n_per_folder": int(args.n_per_folder),
        "seed": int(args.seed),
        "num_folders": len(folders_meta),
        "total_selected": len(sampled_items),
        "folders": folders_meta,
        "samples": sampled_items,
    }

    out_path = (
        Path(args.output_json).expanduser().resolve()
        if args.output_json
        else (root / f"sample_list_n{int(args.n_per_folder)}.json")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({k: out[k] for k in ("root_dir", "num_folders", "total_selected")}, ensure_ascii=False, indent=2))
    print(f"JSON salvo em: {out_path}")


if __name__ == "__main__":
    main()
