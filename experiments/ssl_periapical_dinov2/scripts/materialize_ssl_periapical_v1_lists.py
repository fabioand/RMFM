#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_image_path(row: dict, stem: str, images_dir: Path) -> Path | None:
    # 1) tenta campo explícito (se existir no json)
    for key in ("image_path", "img_path", "path"):
        val = row.get(key)
        if isinstance(val, str) and val:
            p = Path(val)
            if not p.is_absolute():
                p = images_dir / p
            if p.exists() and p.is_file():
                return p.resolve()

    # 2) fallback por stem + extensões conhecidas
    for ext in IMAGE_EXTS:
        p = images_dir / f"{stem}{ext}"
        if p.exists() and p.is_file():
            return p.resolve()
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materializa ssl_periapical_v1_keep/drop/manifest a partir de JSONs de predição."
    )
    parser.add_argument("--predictions-dir", required=True, help="Pasta com JSONs de predição (um por imagem).")
    parser.add_argument("--images-dir", required=True, help="Pasta raiz das imagens.")
    parser.add_argument("--output-dir", required=True, help="Pasta de saída dos artefatos v1.")
    parser.add_argument(
        "--keep-labels",
        default="Periapical,Interproximal",
        help="Labels separadas por vírgula a manter no keep (default: Periapical,Interproximal).",
    )
    parser.add_argument("--progress-every", type=int, default=5000, help="Log de progresso a cada N JSONs.")
    parser.add_argument("--overwrite", action="store_true", help="Permite sobrescrever arquivos finais existentes.")
    parser.add_argument("--dry-run", action="store_true", help="Somente imprime contagens, sem escrever arquivos.")
    args = parser.parse_args()

    predictions_dir = Path(args.predictions_dir)
    images_dir = Path(args.images_dir)
    output_dir = Path(args.output_dir)
    keep_labels = {x.strip() for x in str(args.keep_labels).split(",") if x.strip()}

    if not predictions_dir.exists():
        raise SystemExit(f"--predictions-dir não encontrado: {predictions_dir}")
    if not images_dir.exists():
        raise SystemExit(f"--images-dir não encontrado: {images_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    keep_path = output_dir / "ssl_periapical_v1_keep.txt"
    drop_path = output_dir / "ssl_periapical_v1_drop.txt"
    manifest_path = output_dir / "ssl_periapical_v1_manifest.json"

    if not args.overwrite and not args.dry_run:
        existing = [p for p in (keep_path, drop_path, manifest_path) if p.exists()]
        if existing:
            raise SystemExit(
                "Arquivos de saída já existem e --overwrite não foi informado: "
                + ", ".join(str(p) for p in existing)
            )

    files = sorted([p for p in predictions_dir.glob("*.json") if not p.name.startswith("_")])
    if not files:
        raise SystemExit(f"Nenhum JSON de predição encontrado em: {predictions_dir}")

    keep: list[str] = []
    drop: list[str] = []
    pred_counter: Counter[str] = Counter()
    missing_images = 0
    parse_errors = 0

    total = len(files)
    for i, p in enumerate(files, start=1):
        try:
            row = _load_json(p)
        except Exception:
            parse_errors += 1
            continue

        label = str(row.get("pred_label", "")).strip()
        pred_counter[label] += 1

        stem = p.stem
        img = _resolve_image_path(row=row, stem=stem, images_dir=images_dir)
        if img is None:
            missing_images += 1
            continue

        if label in keep_labels:
            keep.append(str(img))
        else:
            drop.append(str(img))

        if args.progress_every > 0 and (i % args.progress_every == 0 or i == total):
            print(f"[progress] {i}/{total} jsons processados")

    keep = sorted(set(keep))
    drop = sorted(set(drop))
    overlap = sorted(set(keep).intersection(drop))
    if overlap:
        raise SystemExit(f"Inconsistência: {len(overlap)} paths aparecem em keep e drop.")

    manifest = {
        "dataset_id": "ssl_periapical_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_predictions_dir": str(predictions_dir.resolve()),
        "source_images_dir": str(images_dir.resolve()),
        "selection_rule": {"keep_labels": sorted(keep_labels)},
        "counts": {
            "num_json_predictions": int(total),
            "num_keep": int(len(keep)),
            "num_drop": int(len(drop)),
            "num_missing_images": int(missing_images),
            "num_parse_errors": int(parse_errors),
        },
        "pred_label_counts": dict(sorted(pred_counter.items(), key=lambda kv: kv[0])),
        "files": {
            "keep_list": str(keep_path.resolve()),
            "drop_list": str(drop_path.resolve()),
        },
    }

    print("[summary]")
    print(json.dumps(manifest["counts"], ensure_ascii=False, indent=2))
    print("[labels]")
    print(json.dumps(manifest["pred_label_counts"], ensure_ascii=False, indent=2))

    if args.dry_run:
        print("[dry-run] nenhum arquivo foi escrito.")
        return

    keep_path.write_text("\n".join(keep) + ("\n" if keep else ""), encoding="utf-8")
    drop_path.write_text("\n".join(drop) + ("\n" if drop else ""), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[done]")
    print(f"keep: {keep_path}")
    print(f"drop: {drop_path}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()

