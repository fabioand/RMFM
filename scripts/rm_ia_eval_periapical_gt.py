#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests

DEFAULT_BASE_URL = "https://api.radiomemory.com.br/ia-idoc"
DEFAULT_USERNAME = "test"
DEFAULT_PASSWORD = "A)mks8aNKjanm9"
DEFAULT_TIMEOUT = 90
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def load_env_file(env_path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not env_path.exists():
        return out
    for line in env_path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#") or "=" not in raw:
            continue
        k, v = raw.split("=", 1)
        k = k.strip()
        v = v.strip().strip("'").strip('"')
        out[k] = v
    return out


def resolve_auth(args: argparse.Namespace) -> tuple[str, str, str, int]:
    env_file_vars = load_env_file(Path(args.env_file)) if args.env_file else {}
    base_url = args.base_url or os.getenv("RM_BASE_URL") or env_file_vars.get("RM_BASE_URL") or DEFAULT_BASE_URL
    username = args.username or os.getenv("RM_USERNAME") or env_file_vars.get("RM_USERNAME") or DEFAULT_USERNAME
    password = args.password or os.getenv("RM_PASSWORD") or env_file_vars.get("RM_PASSWORD") or DEFAULT_PASSWORD
    timeout = int(args.timeout or os.getenv("RM_TIMEOUT") or env_file_vars.get("RM_TIMEOUT") or DEFAULT_TIMEOUT)
    return base_url, username, password, timeout


def login(base_url: str, username: str, password: str, timeout: int) -> dict[str, str]:
    url = f"{base_url.rstrip('/')}/v1/auth/token"
    headers = {"Content-type": "application/x-www-form-urlencoded", "accept": "application/json"}
    body = f"grant_type=&username={username}&password={password}&scope=&client_id=&client_secret="
    resp = requests.post(url, headers=headers, data=body, timeout=timeout)
    resp.raise_for_status()
    payload = resp.json()
    token_type = payload.get("token_type")
    access_token = payload.get("access_token")
    if not token_type or not access_token:
        raise RuntimeError(f"Falha de autenticacao: {payload}")
    return {
        "Authorization": f"{token_type} {access_token}",
        "Content-type": "application/json",
        "Accept": "application/json",
    }


def read_gt(labels_dir: Path, images_dir: Path) -> list[tuple[str, Path, str]]:
    samples: list[tuple[str, Path, str]] = []
    for jp in sorted(labels_dir.glob("*.json")):
        try:
            payload = json.loads(jp.read_text(encoding="utf-8"))
        except Exception:
            continue
        labels = payload.get("labels")
        if not isinstance(labels, list) or not labels:
            continue
        gt = str(labels[0])
        stem = jp.stem
        img = None
        for ext in IMAGE_EXTS:
            p = images_dir / f"{stem}{ext}"
            if p.exists():
                img = p
                break
        if img is None:
            continue
        samples.append((stem, img, gt))
    return samples


def build_payload(image_path: Path) -> dict[str, Any]:
    return {
        "base64_image": base64.b64encode(image_path.read_bytes()).decode("utf-8"),
        "output_width": 0,
        "output_height": 0,
        "threshold": 0.0,
        "resource": "describe",
        "lang": "pt-br",
        "use_cache": False,
    }


def classify_one(
    *,
    stem: str,
    image_path: Path,
    gt: str,
    base_url: str,
    endpoint: str,
    headers: dict[str, str],
    timeout: int,
    retries: int,
    retry_sleep: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    payload = build_payload(image_path)
    url = f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"

    status: int | None = None
    pred: str | None = None
    score: float | None = None
    error: str | None = None
    body_keys: list[str] = []

    for attempt in range(retries + 1):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=timeout)
            status = int(r.status_code)
            try:
                body = r.json()
            except Exception:
                body = {"raw": r.text[:500]}
            if isinstance(body, dict):
                body_keys = list(body.keys())[:10]
                entities = body.get("entities")
                if isinstance(entities, list) and entities:
                    ent0 = entities[0] if isinstance(entities[0], dict) else {}
                    pred = ent0.get("class_name")
                    score = ent0.get("score")
            # sucesso de transporte; só reintenta em 5xx
            if status >= 500 and attempt < retries:
                time.sleep(retry_sleep * (attempt + 1))
                continue
            break
        except Exception as exc:
            error = str(exc)
            if attempt < retries:
                time.sleep(retry_sleep * (attempt + 1))
                continue
            break

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return {
        "stem": stem,
        "image": str(image_path),
        "gt": gt,
        "pred": pred,
        "score": score,
        "status": status,
        "error": error,
        "latency_ms": elapsed_ms,
        "body_keys": body_keys,
    }


def compute_summary(rows: list[dict[str, Any]], base_url: str, endpoint: str, started_at: float) -> dict[str, Any]:
    total = len(rows)
    status_counts = Counter((r["status"] if r["status"] is not None else "EXC") for r in rows)
    valid = [r for r in rows if r.get("status") == 200 and r.get("pred") is not None]
    correct = [r for r in valid if str(r["pred"]) == str(r["gt"])]

    per_class: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "ok": 0})
    for r in valid:
        c = str(r["gt"])
        per_class[c]["n"] += 1
        if str(r["pred"]) == c:
            per_class[c]["ok"] += 1
    per_class_acc = {k: (v["ok"] / v["n"] if v["n"] else 0.0) for k, v in sorted(per_class.items())}

    conf = Counter((str(r["gt"]), str(r["pred"])) for r in valid if str(r["gt"]) != str(r["pred"]))
    top_confusions = [{"gt": gt, "pred": pred, "count": c} for (gt, pred), c in conf.most_common(40)]

    return {
        "base_url": base_url,
        "endpoint": endpoint,
        "num_samples": total,
        "num_valid_predictions": len(valid),
        "coverage": (len(valid) / total) if total else 0.0,
        "num_correct": len(correct),
        "accuracy": (len(correct) / len(valid)) if valid else 0.0,
        "elapsed_s": (time.perf_counter() - started_at),
        "avg_latency_ms": (sum(float(r["latency_ms"]) for r in rows) / total) if total else 0.0,
        "status_counts": dict(status_counts),
        "per_class_accuracy": per_class_acc,
        "top_confusions": top_confusions,
    }


def save_artifacts(output_dir: Path, rows: list[dict[str, Any]], summary: dict[str, Any], *, suffix: str = "") -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_sorted = sorted(rows, key=lambda x: x["stem"])
    tag = f"_{suffix}" if suffix else ""

    (output_dir / f"rows{tag}.json").write_text(
        json.dumps(rows_sorted, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / f"summary{tag}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / f"top_confusions{tag}.json").write_text(
        json.dumps(summary.get("top_confusions", []), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (output_dir / f"rows{tag}.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["stem", "image", "gt", "pred", "score", "status", "error", "latency_ms"],
        )
        writer.writeheader()
        for r in rows_sorted:
            writer.writerow({k: r.get(k) for k in writer.fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Avalia classificacao RM periapical contra GT (JSON labels) com workers e checkpoint incremental."
    )
    parser.add_argument("--images-dir", default="/Users/fabioandrade/RMFM/Downloads/imgs_class")
    parser.add_argument("--labels-dir", default="/Users/fabioandrade/RMFM/Downloads/periapical_classificacao")
    parser.add_argument("--endpoint", default="v1/periapicals/classification")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--username", default="")
    parser.add_argument("--password", default="")
    parser.add_argument("--timeout", type=int, default=0)
    parser.add_argument("--env-file", default="/Users/fabioandrade/RMFM/.env")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-sleep", type=float, default=0.6)
    parser.add_argument("--max-samples", type=int, default=0, help="0 = usar todos")
    parser.add_argument("--save-every", type=int, default=100, help="checkpoint a cada N concluídos")
    parser.add_argument("--output-dir", default="/Users/fabioandrade/RMFM/out/rm_api_periapical_eval_1312_workers")
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    labels_dir = Path(args.labels_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_url, username, password, timeout = resolve_auth(args)
    if not password:
        raise SystemExit("Sem senha RM. Passe --password ou configure RM_PASSWORD (.env ou env var).")

    samples = read_gt(labels_dir, images_dir)
    if args.max_samples > 0:
        samples = samples[: args.max_samples]
    if not samples:
        raise SystemExit("Nenhum sample GT+imagem encontrado.")

    print(f"Base URL: {base_url}", flush=True)
    print(f"Endpoint: {args.endpoint}", flush=True)
    print(f"Samples: {len(samples)}", flush=True)
    print(f"Workers: {args.workers}", flush=True)

    headers = login(base_url, username, password, timeout)
    print("Login OK", flush=True)

    started_at = time.perf_counter()
    rows: list[dict[str, Any]] = []
    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as ex:
        futures = [
            ex.submit(
                classify_one,
                stem=stem,
                image_path=img,
                gt=gt,
                base_url=base_url,
                endpoint=args.endpoint,
                headers=headers,
                timeout=timeout,
                retries=max(0, int(args.retries)),
                retry_sleep=float(args.retry_sleep),
            )
            for stem, img, gt in samples
        ]

        done = 0
        total = len(futures)
        for fut in as_completed(futures):
            row = fut.result()
            with lock:
                rows.append(row)
                done += 1
                if done % 25 == 0 or done == total:
                    print(f"{done}/{total}", flush=True)
                if args.save_every > 0 and (done % int(args.save_every) == 0):
                    partial = compute_summary(rows, base_url, args.endpoint, started_at)
                    save_artifacts(output_dir, rows, partial, suffix="partial")

    summary = compute_summary(rows, base_url, args.endpoint, started_at)
    save_artifacts(output_dir, rows, summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"OUT_DIR: {output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
