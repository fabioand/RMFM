#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

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

    subprocess.run(cmd, check=True)
    return output_dir / "e2_retrain_head" / "summary.json"


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
        "baseline_run_dir": str(Path(args.baseline_run_dir).resolve()),
    }
    _save_json(out_dir / "downstream_eval_manifest.json", meta)
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

