#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

from dino_v3_lab import (
    extract_global_embedding,
    extract_global_embedding_and_cls_patch_map,
    load_backbone,
    resolve_device,
    resolve_hf_token,
)

VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def iter_images(folder: Path) -> list[Path]:
    return sorted(
        p
        for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in VALID_EXT
    )


def _preprocess_signature(processor_kwargs: dict[str, Any]) -> str:
    if not processor_kwargs:
        return "default"
    return json.dumps(processor_kwargs, sort_keys=True, ensure_ascii=True)


def _build_cache_key(
    image_path: Path,
    model_id: str,
    preprocess_sig: str,
) -> str:
    st = image_path.stat()
    raw = "||".join(
        [
            str(image_path.resolve()),
            str(st.st_size),
            str(st.st_mtime_ns),
            model_id,
            preprocess_sig,
        ]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _load_cached_embedding(cache_dir: Path, key: str) -> np.ndarray | None:
    emb_path = cache_dir / f"{key}.npy"
    if not emb_path.exists():
        return None
    try:
        arr = np.load(emb_path)
        if arr.ndim != 1:
            return None
        return arr.astype(np.float32, copy=False)
    except Exception:
        return None


def _save_cached_embedding(cache_dir: Path, key: str, emb: np.ndarray) -> None:
    emb_path = cache_dir / f"{key}.npy"
    np.save(emb_path, emb.astype(np.float32, copy=False))


def _load_cached_attention(attention_cache_dir: Path, key: str) -> np.ndarray | None:
    att_path = attention_cache_dir / f"{key}.npy"
    if not att_path.exists():
        return None
    try:
        arr = np.load(att_path)
        if arr.ndim != 2:
            return None
        return arr.astype(np.float32, copy=False)
    except Exception:
        return None


def _save_cached_attention(attention_cache_dir: Path, key: str, att: np.ndarray) -> None:
    att_path = attention_cache_dir / f"{key}.npy"
    np.save(att_path, att.astype(np.float32, copy=False))


def _resize_attention_to_original_aspect(
    att: np.ndarray,
    orig_w: int,
    orig_h: int,
    processed_min_dim: int,
) -> np.ndarray:
    if orig_w <= 0 or orig_h <= 0 or processed_min_dim <= 0:
        return att

    if orig_w >= orig_h:
        tgt_h = processed_min_dim
        tgt_w = max(1, int(round((orig_w / orig_h) * processed_min_dim)))
    else:
        tgt_w = processed_min_dim
        tgt_h = max(1, int(round((orig_h / orig_w) * processed_min_dim)))

    src = np.clip(att, 0.0, 1.0).astype(np.float32)
    src_u8 = (src * 255.0).astype(np.uint8)
    resized_u8 = np.array(
        Image.fromarray(src_u8).resize((tgt_w, tgt_h), resample=Image.BILINEAR),
        dtype=np.uint8,
    )
    return resized_u8.astype(np.float32) / 255.0


def _save_attention_png(att: np.ndarray, out_path: Path, orig_w: int, orig_h: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    processed_min_dim = int(min(att.shape[0], att.shape[1]))
    x = _resize_attention_to_original_aspect(att, orig_w=orig_w, orig_h=orig_h, processed_min_dim=processed_min_dim)
    x = np.clip(x, 0.0, 1.0).astype(np.float32)
    # Turbo-like control points for clearer visual separation of low/high attention.
    stops = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], dtype=np.float32)
    colors = np.array(
        [
            [48, 18, 59],    # deep purple
            [50, 87, 178],   # blue
            [23, 168, 220],  # cyan
            [122, 209, 81],  # green
            [245, 190, 39],  # yellow/orange
            [180, 4, 38],    # red
        ],
        dtype=np.float32,
    )

    flat = x.reshape(-1)
    r = np.interp(flat, stops, colors[:, 0]).reshape(x.shape)
    g = np.interp(flat, stops, colors[:, 1]).reshape(x.shape)
    b = np.interp(flat, stops, colors[:, 2]).reshape(x.shape)
    rgb = np.stack([r, g, b], axis=-1).astype(np.uint8)
    Image.fromarray(rgb).save(out_path)


def _render_report(title: str, rows: list[dict[str, Any]]) -> str:
    by_cluster: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_cluster.setdefault(int(row["cluster"]), []).append(row)

    sections = []
    for cluster_id, items in sorted(by_cluster.items(), key=lambda kv: len(kv[1])):
        items_sorted = sorted(items, key=lambda r: float(r["dist_to_centroid"]))
        cards = []
        for item in items_sorted:
            image_uri = str(item.get("image_src", Path(item["image"]).resolve().as_uri()))
            attention_map = item.get("attention_map", "")
            attention_src = str(item.get("attention_src", ""))
            inf_ms = item.get("inference_ms")
            emb_source = item.get("embedding_source", "-")
            inf_line = (
                f'<p><b>Inferência:</b> {float(inf_ms):.1f} ms</p>'
                if isinstance(inf_ms, (int, float))
                else ""
            )
            if attention_map:
                att_uri = attention_src if attention_src else Path(attention_map).resolve().as_uri()
                media_html = (
                    '<div class="thumb dual">'
                    f'<div class="pane"><img src="{image_uri}" alt="{html.escape(item["name"])}" loading="lazy" /><span>Imagem</span></div>'
                    f'<div class="pane"><img src="{att_uri}" alt="attention map" loading="lazy" /><span>Attention</span></div>'
                    "</div>"
                )
            else:
                media_html = (
                    f'<div class="thumb"><img src="{image_uri}" alt="{html.escape(item["name"])}" loading="lazy" /></div>'
                )
            cards.append(
                f"""
                <article class="card">
                  {media_html}
                  <div class="body">
                    <h3 title="{html.escape(item['image'])}">{html.escape(item['name'])}</h3>
                    <p><b>Cluster:</b> {item['cluster']}</p>
                    <p><b>Dist. centroide:</b> {item['dist_to_centroid']:.4f}</p>
                    <p><b>Fonte:</b> {html.escape(str(emb_source))}</p>
                    {inf_line}
                  </div>
                </article>
                """
            )

        sections.append(
            f"""
            <section class="cluster">
              <h2>Cluster {cluster_id} ({len(items)} imagens)</h2>
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
      --card: #ffffff;
      --line: #d1d9e6;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: linear-gradient(180deg, #edf2f7 0%, var(--bg) 40%);
    }}
    .wrap {{ max-width: 1500px; margin: 0 auto; padding: 24px; }}
    h1 {{ margin: 0 0 14px; font-size: 24px; }}
    .cluster {{ margin-top: 20px; }}
    h2 {{ margin: 0 0 10px; font-size: 18px; color: #0f172a; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(230px, 1fr));
      gap: 12px;
    }}
    .card {{
      border: 1px solid var(--line);
      border-radius: 12px;
      overflow: hidden;
      background: var(--card);
      box-shadow: 0 4px 12px rgba(15, 23, 42, 0.06);
    }}
    .thumb {{ height: 170px; background: #eef2f7; }}
    .thumb img {{ width: 100%; height: 100%; object-fit: contain; }}
    .thumb.dual {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1px;
      background: #d6dce6;
    }}
    .thumb.dual .pane {{
      position: relative;
      background: #eef2f7;
    }}
    .thumb.dual .pane span {{
      position: absolute;
      left: 6px;
      bottom: 6px;
      font-size: 10px;
      color: #0f172a;
      background: rgba(255,255,255,0.8);
      padding: 2px 4px;
      border-radius: 4px;
    }}
    .body {{ padding: 10px 11px 12px; }}
    .body h3 {{
      margin: 0 0 8px;
      font-size: 12px;
      line-height: 1.35;
      word-break: break-all;
      color: #0f172a;
    }}
    .body p {{
      margin: 4px 0;
      font-size: 12px;
      color: var(--muted);
    }}
  </style>
</head>
<body>
  <main class="wrap">
    <h1>{html.escape(title)}</h1>
    {''.join(sections)}
  </main>
</body>
</html>
"""


def _ensure_readable_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path, 0o755)
    except Exception:
        pass


def _relative_web_src(path: Path, base_dir: Path) -> str:
    try:
        # Important: do not call resolve() on symlinks here.
        # If resolved, link paths point to external targets (e.g. /dataminer)
        # and we lose the relative URL under output_dir.
        return path.absolute().relative_to(base_dir.absolute()).as_posix()
    except Exception:
        return path.absolute().as_uri()


def _make_image_symlink(src_image: Path, images_dir: Path, linked_images_dir: Path) -> Path:
    rel = src_image.resolve().relative_to(images_dir.resolve())
    link_path = linked_images_dir / rel
    _ensure_readable_dir(link_path.parent)
    if link_path.exists() or link_path.is_symlink():
        try:
            if link_path.is_symlink() and link_path.resolve() == src_image.resolve():
                return link_path
            link_path.unlink()
        except Exception:
            pass
    link_path.symlink_to(src_image.resolve())
    return link_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Clusteriza imagens via embeddings DINO e gera relatório HTML")
    parser.add_argument("--model-id", default="facebook/dinov2-small")
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--output-dir", default="outputs/cluster_report")
    parser.add_argument("--n-clusters", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0, help="0 = usar todas as imagens encontradas")
    parser.add_argument("--shortest-edge", type=int, default=0, help="Resize shortest edge no preprocess (0 = padrão do modelo)")
    parser.add_argument("--crop-size", type=int, default=0, help="Center crop quadrado no preprocess (0 = padrão do modelo)")
    parser.add_argument("--cache-dir", default="outputs/embedding_cache", help="Pasta do cache de embeddings")
    parser.add_argument("--attention-cache-dir", default="outputs/attention_cache", help="Pasta do cache de mapas de atenção")
    parser.add_argument("--no-cache", action="store_true", help="Desativa leitura/gravação de cache")
    parser.add_argument("--save-attention-maps", action="store_true", help="Salva mapa CLS·patch por imagem")
    parser.add_argument("--attention-dir", default="", help="Pasta de saída dos PNGs de atenção (default: <output-dir>/attention_maps)")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--hf-token-env", default="HF_TOKEN")
    parser.add_argument("--offline", action="store_true", help="Usa apenas cache local do Hugging Face")
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    if not images_dir.exists():
        raise SystemExit(f"Pasta não encontrada: {images_dir}")

    output_dir = Path(args.output_dir)
    _ensure_readable_dir(output_dir)
    cache_dir = Path(args.cache_dir)
    attention_cache_dir = Path(args.attention_cache_dir)
    use_cache = not args.no_cache
    save_attention_maps = bool(args.save_attention_maps)
    attention_dir = Path(args.attention_dir) if args.attention_dir else (output_dir / "attention_maps")
    if use_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        if save_attention_maps:
            attention_cache_dir.mkdir(parents=True, exist_ok=True)
    if save_attention_maps:
        _ensure_readable_dir(attention_dir)

    linked_images_dir = output_dir / "_linked_images"
    _ensure_readable_dir(linked_images_dir)

    image_paths = iter_images(images_dir)
    if args.limit > 0:
        image_paths = image_paths[: args.limit]
    if not image_paths:
        raise SystemExit("Nenhuma imagem encontrada.")

    n_clusters = max(2, min(args.n_clusters, len(image_paths)))

    device = "cpu" if args.cpu else resolve_device()
    hf_token = resolve_hf_token(args.hf_token, args.hf_token_env)
    t_model_start = time.perf_counter()
    processor, model = load_backbone(
        args.model_id,
        device=device,
        hf_token=hf_token,
        local_files_only=args.offline,
    )
    model_load_s = time.perf_counter() - t_model_start

    print(f"Modelo: {args.model_id}")
    print(f"Device: {device}")
    print(f"Imagens: {len(image_paths)}")

    processor_kwargs: dict[str, Any] = {}
    if args.shortest_edge > 0:
        processor_kwargs["size"] = {"shortest_edge": int(args.shortest_edge)}
    if args.crop_size > 0:
        processor_kwargs["crop_size"] = {
            "height": int(args.crop_size),
            "width": int(args.crop_size),
        }
    if processor_kwargs:
        print(f"Preprocess override: {processor_kwargs}")
    print(f"Cache: {'ON' if use_cache else 'OFF'} ({cache_dir.resolve()})")
    print(f"Attention maps: {'ON' if save_attention_maps else 'OFF'}")
    if save_attention_maps:
        print(f"Attention dir: {attention_dir.resolve()}")

    print("Extraindo embeddings...")

    embeddings: list[np.ndarray] = []
    inference_ms: list[float] = []
    embedding_source: list[str] = []
    attention_paths: list[str] = []
    cache_hits = 0
    cache_misses = 0
    attention_cache_hits = 0
    attention_cache_misses = 0
    preprocess_sig = _preprocess_signature(processor_kwargs)
    for idx, path in enumerate(image_paths, start=1):
        t0 = time.perf_counter()
        emb = None
        att_map = None
        src = "model"
        key = ""
        if use_cache:
            key = _build_cache_key(path, args.model_id, preprocess_sig)
            emb = _load_cached_embedding(cache_dir, key)
            if save_attention_maps:
                att_map = _load_cached_attention(attention_cache_dir, key)
                if att_map is not None:
                    attention_cache_hits += 1
            if emb is not None and (not save_attention_maps or att_map is not None):
                src = "cache"
                cache_hits += 1
        img_size = None
        if emb is None or (save_attention_maps and att_map is None):
            src = "model"
            img = Image.open(path).convert("RGB")
            img_size = img.size
            if save_attention_maps:
                emb, att_map = extract_global_embedding_and_cls_patch_map(
                    processor,
                    model,
                    img,
                    device,
                    processor_kwargs=processor_kwargs,
                )
                attention_cache_misses += 1
            else:
                emb = extract_global_embedding(
                    processor,
                    model,
                    img,
                    device,
                    processor_kwargs=processor_kwargs,
                )
            if use_cache:
                _save_cached_embedding(cache_dir, key, emb)
                if save_attention_maps and att_map is not None:
                    _save_cached_attention(attention_cache_dir, key, att_map)
            cache_misses += 1

        dt_ms = (time.perf_counter() - t0) * 1000.0
        embeddings.append(emb)
        inference_ms.append(dt_ms)
        embedding_source.append(src)
        if save_attention_maps and att_map is not None:
            if img_size is None:
                with Image.open(path) as img_meta:
                    img_size = img_meta.size
            orig_w, orig_h = int(img_size[0]), int(img_size[1])
            att_name = f"{idx:05d}_{path.stem}.png"
            att_path = attention_dir / att_name
            _save_attention_png(att_map, att_path, orig_w=orig_w, orig_h=orig_h)
            attention_paths.append(str(att_path.resolve()))
        else:
            attention_paths.append("")
        if idx % 10 == 0 or idx == len(image_paths):
            print(f"  {idx}/{len(image_paths)}")

    matrix = np.stack(embeddings, axis=0)

    print("Clusterizando...")
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(matrix)
    centers = km.cluster_centers_

    rows: list[dict[str, Any]] = []
    for i, path in enumerate(image_paths):
        c = int(labels[i])
        dist = float(np.linalg.norm(matrix[i] - centers[c]))
        link_path = _make_image_symlink(path, images_dir=images_dir, linked_images_dir=linked_images_dir)
        image_src = _relative_web_src(link_path, output_dir)
        attention_src = ""
        if attention_paths[i]:
            attention_src = _relative_web_src(Path(attention_paths[i]), output_dir)
        rows.append(
            {
                "image": str(path.resolve()),
                "image_src": image_src,
                "name": path.name,
                "cluster": c,
                "dist_to_centroid": dist,
                "inference_ms": float(inference_ms[i]),
                "embedding_source": embedding_source[i],
                "attention_map": attention_paths[i],
                "attention_src": attention_src,
            }
        )

    inf_arr = np.array(inference_ms, dtype=np.float64)
    timing = {
        "model_load_s": float(model_load_s),
        "num_images": int(len(inference_ms)),
        "total_inference_s": float(inf_arr.sum() / 1000.0),
        "mean_inference_ms": float(inf_arr.mean()),
        "median_inference_ms": float(np.percentile(inf_arr, 50)),
        "p95_inference_ms": float(np.percentile(inf_arr, 95)),
        "min_inference_ms": float(inf_arr.min()),
        "max_inference_ms": float(inf_arr.max()),
        "cache_enabled": bool(use_cache),
        "cache_dir": str(cache_dir.resolve()) if use_cache else "",
        "cache_hits": int(cache_hits),
        "cache_misses": int(cache_misses),
        "attention_maps_enabled": bool(save_attention_maps),
        "attention_dir": str(attention_dir.resolve()) if save_attention_maps else "",
        "attention_cache_hits": int(attention_cache_hits),
        "attention_cache_misses": int(attention_cache_misses),
    }

    summary = {
        "model_id": args.model_id,
        "device": device,
        "num_images": len(image_paths),
        "embedding_dim": int(matrix.shape[1]),
        "n_clusters": int(n_clusters),
        "preprocess": processor_kwargs if processor_kwargs else "default",
        "cluster_sizes": {
            str(c): int(sum(1 for r in rows if int(r["cluster"]) == c))
            for c in sorted(set(int(r["cluster"]) for r in rows))
        },
        "timing": timing,
        "linked_images_dir": str(linked_images_dir.resolve()),
    }

    np.save(output_dir / "embeddings.npy", matrix)
    (output_dir / "cluster_rows.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "timing.json").write_text(
        json.dumps(timing, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    html_path = output_dir / "cluster_report.html"
    html_path.write_text(
        _render_report(
            title=f"DINO Clustering - {args.model_id} ({len(image_paths)} imagens)",
            rows=rows,
        ),
        encoding="utf-8",
    )

    print("\nResumo:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nRelatório HTML: {html_path.resolve()}")


if __name__ == "__main__":
    main()
