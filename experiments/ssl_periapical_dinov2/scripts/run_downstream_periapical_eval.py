#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier

REPO = Path(__file__).resolve().parents[3]
PERI_SRC = REPO / "experiments" / "periapical_dino_classifier" / "src"
if str(PERI_SRC) not in sys.path:
    sys.path.insert(0, str(PERI_SRC))

from dino_periapical_cls.data import PeriapicalDataset, build_label_mapping, discover_samples, stratified_split
from dino_periapical_cls.model import FrozenDinoClassifier
from dino_periapical_cls.train_cached import HeadClassifier, _extract_split_features
from transformers import AutoImageProcessor


def _save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_e1_reuse_old_head(
    model_id: str,
    baseline_run_dir: Path,
    images_dir: Path,
    labels_dir: Path,
    output_dir: Path,
    device: str,
    offline: bool,
) -> Path:
    ckpt = torch.load(baseline_run_dir / "best_head_only.pt", map_location=device)
    old_args = ckpt.get("args", {})
    seed = int(old_args.get("seed", 42))
    val_size = float(old_args.get("val_size", 0.15))
    test_size = float(old_args.get("test_size", 0.15))
    shortest_edge = int(old_args.get("shortest_edge", 256))
    crop_size = int(old_args.get("crop_size", 256))
    feature_batch_size = int(old_args.get("feature_batch_size", 64))
    num_workers = int(old_args.get("num_workers", 0))
    dropout = float(old_args.get("dropout", 0.1))

    samples = discover_samples(images_dir, labels_dir)
    label_to_idx = build_label_mapping(samples)
    splits = stratified_split(samples, val_size=val_size, test_size=test_size, seed=seed)
    ds_test = PeriapicalDataset(splits["test"], label_to_idx)

    processor = AutoImageProcessor.from_pretrained(model_id, local_files_only=offline)
    backbone = FrozenDinoClassifier(
        model_id=model_id,
        num_classes=len(label_to_idx),
        local_files_only=offline,
        freeze_backbone=True,
        dropout=dropout,
    ).to(device)
    backbone.eval()

    x_test, y_test = _extract_split_features(
        backbone=backbone,
        ds=ds_test,
        processor=processor,
        batch_size=feature_batch_size,
        num_workers=num_workers,
        shortest_edge=shortest_edge,
        crop_size=crop_size,
        device=device,
    )

    head = HeadClassifier(in_dim=int(ckpt["in_dim"]), num_classes=len(label_to_idx), dropout=dropout).to(device)
    head.load_state_dict(ckpt["head_state_dict"], strict=True)
    head.eval()
    xb = torch.from_numpy(x_test).to(device)
    with torch.no_grad():
        logits = head(xb).detach().cpu().numpy()
    pred = np.argmax(logits, axis=1)
    metrics = {
        "test_accuracy": float(accuracy_score(y_test, pred)),
        "test_macro_f1": float(f1_score(y_test, pred, average="macro")),
        "num_test": int(len(y_test)),
    }
    out = output_dir / "e1_reuse_head_summary.json"
    _save_json(out, metrics)
    return out


def run_e2_retrain_head(
    model_id: str,
    images_dir: Path,
    labels_dir: Path,
    output_dir: Path,
    baseline_run_dir: Path,
    offline: bool,
    cpu: bool,
) -> Path:
    ckpt = torch.load(baseline_run_dir / "best_head_only.pt", map_location="cpu")
    old_args = ckpt.get("args", {})
    cmd = [
        sys.executable,
        str(REPO / "experiments" / "periapical_dino_classifier" / "scripts" / "train_frozen_head_cached.py"),
        "--images-dir",
        str(images_dir),
        "--labels-dir",
        str(labels_dir),
        "--output-dir",
        str(output_dir / "e2_retrain_head"),
        "--model-id",
        str(model_id),
        "--epochs",
        str(int(old_args.get("epochs", 60))),
        "--batch-size",
        str(int(old_args.get("batch_size", 128))),
        "--feature-batch-size",
        str(int(old_args.get("feature_batch_size", 64))),
        "--lr",
        str(float(old_args.get("lr", 1e-3))),
        "--weight-decay",
        str(float(old_args.get("weight_decay", 1e-4))),
        "--dropout",
        str(float(old_args.get("dropout", 0.1))),
        "--shortest-edge",
        str(int(old_args.get("shortest_edge", 256))),
        "--crop-size",
        str(int(old_args.get("crop_size", 256))),
        "--val-size",
        str(float(old_args.get("val_size", 0.15))),
        "--test-size",
        str(float(old_args.get("test_size", 0.15))),
        "--num-workers",
        str(int(old_args.get("num_workers", 0))),
        "--seed",
        str(int(old_args.get("seed", 42))),
    ]
    if bool(old_args.get("augment_flip_mirror", False)):
        cmd.append("--augment-flip-mirror")
    if offline:
        cmd.append("--offline")
    if cpu:
        cmd.append("--cpu")

    env = dict(os.environ)
    existing_pythonpath = env.get("PYTHONPATH", "").strip()
    peri_src_str = str(PERI_SRC)
    if existing_pythonpath:
        env["PYTHONPATH"] = f"{peri_src_str}:{existing_pythonpath}"
    else:
        env["PYTHONPATH"] = peri_src_str

    subprocess.run(cmd, check=True, cwd=str(REPO), env=env)
    return output_dir / "e2_retrain_head" / "summary.json"


def run_e2_retrain_knn(
    model_id: str,
    images_dir: Path,
    labels_dir: Path,
    output_dir: Path,
    baseline_run_dir: Path,
    offline: bool,
    cpu: bool,
    k_candidates: list[int],
) -> Path:
    ckpt = torch.load(baseline_run_dir / "best_head_only.pt", map_location="cpu")
    old_args = ckpt.get("args", {})
    seed = int(old_args.get("seed", 42))
    val_size = float(old_args.get("val_size", 0.15))
    test_size = float(old_args.get("test_size", 0.15))
    shortest_edge = int(old_args.get("shortest_edge", 256))
    crop_size = int(old_args.get("crop_size", 256))
    feature_batch_size = int(old_args.get("feature_batch_size", 64))
    num_workers = int(old_args.get("num_workers", 0))
    dropout = float(old_args.get("dropout", 0.1))
    use_flip_mirror = bool(old_args.get("augment_flip_mirror", False))

    output_knn_dir = output_dir / "e2_retrain_knn"
    output_knn_dir.mkdir(parents=True, exist_ok=True)
    features_dir = output_knn_dir / "features_cache"
    features_dir.mkdir(parents=True, exist_ok=True)

    device = "cpu"
    if not cpu and torch.cuda.is_available():
        device = "cuda"
    elif not cpu and torch.backends.mps.is_available():
        device = "mps"

    samples = discover_samples(images_dir, labels_dir)
    label_to_idx = build_label_mapping(samples)
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    splits = stratified_split(samples, val_size=val_size, test_size=test_size, seed=seed)

    ds_train = PeriapicalDataset(splits["train"], label_to_idx)
    ds_val = PeriapicalDataset(splits["val"], label_to_idx)
    ds_test = PeriapicalDataset(splits["test"], label_to_idx)

    processor = AutoImageProcessor.from_pretrained(model_id, local_files_only=offline)
    backbone = FrozenDinoClassifier(
        model_id=model_id,
        num_classes=len(label_to_idx),
        local_files_only=offline,
        freeze_backbone=True,
        dropout=dropout,
    ).to(device)
    backbone.eval()

    x_train, y_train = _extract_split_features(
        backbone=backbone,
        ds=ds_train,
        processor=processor,
        batch_size=feature_batch_size,
        num_workers=num_workers,
        shortest_edge=shortest_edge,
        crop_size=crop_size,
        device=device,
        augment_flip_mirror=use_flip_mirror,
        mirror_label_idx_map=None,
    )
    x_val, y_val = _extract_split_features(
        backbone=backbone,
        ds=ds_val,
        processor=processor,
        batch_size=feature_batch_size,
        num_workers=num_workers,
        shortest_edge=shortest_edge,
        crop_size=crop_size,
        device=device,
    )
    x_test, y_test = _extract_split_features(
        backbone=backbone,
        ds=ds_test,
        processor=processor,
        batch_size=feature_batch_size,
        num_workers=num_workers,
        shortest_edge=shortest_edge,
        crop_size=crop_size,
        device=device,
    )

    np.save(features_dir / "x_train.npy", x_train)
    np.save(features_dir / "y_train.npy", y_train)
    np.save(features_dir / "x_val.npy", x_val)
    np.save(features_dir / "y_val.npy", y_val)
    np.save(features_dir / "x_test.npy", x_test)
    np.save(features_dir / "y_test.npy", y_test)

    candidates: list[dict[str, float]] = []
    best_k = None
    best_val_f1 = -1.0
    best_model = None
    for k in k_candidates:
        model = KNeighborsClassifier(n_neighbors=int(k), weights="distance", metric="cosine")
        model.fit(x_train, y_train)
        val_pred = model.predict(x_val)
        val_acc = float(accuracy_score(y_val, val_pred))
        val_f1 = float(f1_score(y_val, val_pred, average="macro"))
        candidates.append({"k": int(k), "val_accuracy": val_acc, "val_macro_f1": val_f1})
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_k = int(k)
            best_model = model

    if best_model is None or best_k is None:
        raise RuntimeError("Falha ao ajustar KNN (nenhum candidato válido).")

    test_pred = best_model.predict(x_test)
    test_acc = float(accuracy_score(y_test, test_pred))
    test_f1 = float(f1_score(y_test, test_pred, average="macro"))

    summary = {
        "mode": "cached_features_knn",
        "model_id": model_id,
        "device": device,
        "num_samples": int(len(samples)),
        "num_classes": int(len(label_to_idx)),
        "feature_dim": int(x_train.shape[1]) if x_train.ndim == 2 else 0,
        "augment_flip_mirror": bool(use_flip_mirror),
        "best_k": int(best_k),
        "best_val_macro_f1": float(best_val_f1),
        "val_candidates": candidates,
        "test": {
            "accuracy": test_acc,
            "macro_f1": test_f1,
        },
        "paths": {
            "features_dir": str(features_dir.resolve()),
            "summary": str((output_knn_dir / "summary.json").resolve()),
        },
    }
    _save_json(output_knn_dir / "summary.json", summary)
    _save_json(output_knn_dir / "label_to_idx.json", label_to_idx)
    _save_json(output_knn_dir / "idx_to_label.json", {str(k): v for k, v in idx_to_label.items()})
    return output_knn_dir / "summary.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Roda avaliacao downstream E1/E2 para backbone SSL exportado")
    parser.add_argument("--backbone-dir", required=True)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--labels-dir", required=True)
    parser.add_argument(
        "--baseline-run-dir",
        default=str(REPO / "experiments/periapical_dino_classifier/outputs/run_cached_head_256_flipmirror_v1"),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--e2-classifier", choices=("mlp", "knn"), default="mlp")
    parser.add_argument("--knn-k-candidates", default="1,3,5,7,9,11,15,21")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cpu"
    if not args.cpu and torch.cuda.is_available():
        device = "cuda"
    elif not args.cpu and torch.backends.mps.is_available():
        device = "mps"

    e1 = run_e1_reuse_old_head(
        model_id=str(args.backbone_dir),
        baseline_run_dir=Path(args.baseline_run_dir),
        images_dir=Path(args.images_dir),
        labels_dir=Path(args.labels_dir),
        output_dir=out_dir,
        device=device,
        offline=bool(args.offline),
    )
    if args.e2_classifier == "knn":
        k_candidates = [int(x.strip()) for x in str(args.knn_k_candidates).split(",") if x.strip()]
        e2 = run_e2_retrain_knn(
            model_id=str(args.backbone_dir),
            images_dir=Path(args.images_dir),
            labels_dir=Path(args.labels_dir),
            output_dir=out_dir,
            baseline_run_dir=Path(args.baseline_run_dir),
            offline=bool(args.offline),
            cpu=bool(args.cpu),
            k_candidates=k_candidates,
        )
    else:
        e2 = run_e2_retrain_head(
            model_id=str(args.backbone_dir),
            images_dir=Path(args.images_dir),
            labels_dir=Path(args.labels_dir),
            output_dir=out_dir,
            baseline_run_dir=Path(args.baseline_run_dir),
            offline=bool(args.offline),
            cpu=bool(args.cpu),
        )

    meta = {
        "e1_summary": str(e1.resolve()),
        "e2_summary": str(e2.resolve()),
        "e2_classifier": str(args.e2_classifier),
        "baseline_run_dir": str(Path(args.baseline_run_dir).resolve()),
    }
    _save_json(out_dir / "downstream_eval_manifest.json", meta)
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
