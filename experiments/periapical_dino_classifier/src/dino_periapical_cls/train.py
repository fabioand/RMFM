from __future__ import annotations

import json
import os
import random
from argparse import Namespace
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover
    SummaryWriter = None
from transformers import AutoImageProcessor

from .data import (
    PeriapicalDataset,
    build_label_mapping,
    discover_samples,
    make_collate_fn,
    stratified_split,
)
from .model import FrozenDinoClassifier, count_trainable_params


def resolve_device(prefer_mps: bool = True) -> str:
    if torch.cuda.is_available():
        return "cuda"
    if prefer_mps and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_hf_token(explicit_token: str | None = None, token_env: str = "HF_TOKEN") -> str | None:
    if explicit_token:
        return explicit_token
    return os.environ.get(token_env)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _batch_to_device(batch: dict[str, Any], device: str) -> tuple[torch.Tensor, torch.Tensor, list[str], list[str]]:
    pixel_values = batch["pixel_values"].to(device)
    labels = torch.tensor(batch["labels"], dtype=torch.long, device=device)
    paths = batch["paths"]
    stems = batch["stems"]
    return pixel_values, labels, paths, stems


@torch.no_grad()
def evaluate(
    model: FrozenDinoClassifier,
    loader: DataLoader,
    device: str,
) -> dict[str, Any]:
    model.eval()

    losses: list[float] = []
    y_true: list[int] = []
    y_pred: list[int] = []

    for batch in loader:
        pixel_values, labels, _, _ = _batch_to_device(batch, device)
        logits = model(pixel_values)
        loss = F.cross_entropy(logits, labels)
        preds = torch.argmax(logits, dim=1)

        losses.append(float(loss.item()))
        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(preds.detach().cpu().tolist())

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "y_true": y_true,
        "y_pred": y_pred,
    }


def train_one_epoch(
    model: FrozenDinoClassifier,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> dict[str, float]:
    model.train()
    if model.freeze_backbone:
        model.backbone.eval()

    losses: list[float] = []
    y_true: list[int] = []
    y_pred: list[int] = []

    for batch in loader:
        pixel_values, labels, _, _ = _batch_to_device(batch, device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(pixel_values)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()

        preds = torch.argmax(logits, dim=1)
        losses.append(float(loss.item()))
        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(preds.detach().cpu().tolist())

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
    }


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _class_distribution(samples, name: str) -> dict[str, int]:
    c = Counter(s.label_name for s in samples)
    print(f"{name}: {len(samples)} amostras | classes={len(c)}")
    return dict(sorted(c.items(), key=lambda kv: kv[0]))


def run_training(args: Namespace) -> None:
    set_seed(int(args.seed))

    images_dir = Path(args.images_dir)
    labels_dir = Path(args.labels_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = output_dir / "tb"

    samples = discover_samples(images_dir, labels_dir)
    if len(samples) == 0:
        raise SystemExit("Nenhuma amostra válida encontrada (imagem + label).")

    if int(args.max_samples) > 0:
        samples = samples[: int(args.max_samples)]

    label_to_idx = build_label_mapping(samples)
    idx_to_label = {v: k for k, v in label_to_idx.items()}

    splits = stratified_split(
        samples,
        val_size=float(args.val_size),
        test_size=float(args.test_size),
        seed=int(args.seed),
    )

    split_stats = {
        "train": _class_distribution(splits["train"], "train"),
        "val": _class_distribution(splits["val"], "val"),
        "test": _class_distribution(splits["test"], "test"),
    }

    hf_token = resolve_hf_token(args.hf_token, args.hf_token_env)
    processor = AutoImageProcessor.from_pretrained(
        args.model_id,
        token=hf_token,
        local_files_only=bool(args.offline),
    )

    collate_fn = make_collate_fn(
        processor,
        shortest_edge=int(args.shortest_edge),
        crop_size=int(args.crop_size),
    )

    train_ds = PeriapicalDataset(splits["train"], label_to_idx)
    val_ds = PeriapicalDataset(splits["val"], label_to_idx)
    test_ds = PeriapicalDataset(splits["test"], label_to_idx)

    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        collate_fn=collate_fn,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        collate_fn=collate_fn,
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        collate_fn=collate_fn,
        pin_memory=False,
    )

    device = "cpu" if bool(args.cpu) else resolve_device()
    model = FrozenDinoClassifier(
        model_id=args.model_id,
        num_classes=len(label_to_idx),
        hf_token=hf_token,
        local_files_only=bool(args.offline),
        freeze_backbone=True,
        dropout=float(args.dropout),
    ).to(device)

    print(f"device={device}")
    print(f"trainable_params={count_trainable_params(model)}")

    writer = None
    if not bool(args.no_tensorboard):
        if SummaryWriter is None:
            print("TensorBoard indisponível neste ambiente (instale `tensorboard`). Seguindo sem logs TB.")
        else:
            writer = SummaryWriter(log_dir=str(tb_dir))
            writer.add_text("run/model_id", str(args.model_id))
            writer.add_text("run/device", str(device))
            writer.add_text("run/output_dir", str(output_dir.resolve()))

    optimizer = torch.optim.AdamW(
        params=[p for p in model.parameters() if p.requires_grad],
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )

    history: list[dict[str, Any]] = []
    best_val_f1 = -1.0
    best_epoch = -1
    best_ckpt_path = output_dir / "best_head.pt"

    for epoch in range(1, int(args.epochs) + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)

        row = {
            "epoch": epoch,
            "train": {
                "loss": train_metrics["loss"],
                "accuracy": train_metrics["accuracy"],
                "macro_f1": train_metrics["macro_f1"],
            },
            "val": {
                "loss": val_metrics["loss"],
                "accuracy": val_metrics["accuracy"],
                "macro_f1": val_metrics["macro_f1"],
            },
        }
        history.append(row)

        if writer is not None:
            writer.add_scalar("train/loss", train_metrics["loss"], epoch)
            writer.add_scalar("train/accuracy", train_metrics["accuracy"], epoch)
            writer.add_scalar("train/macro_f1", train_metrics["macro_f1"], epoch)
            writer.add_scalar("val/loss", val_metrics["loss"], epoch)
            writer.add_scalar("val/accuracy", val_metrics["accuracy"], epoch)
            writer.add_scalar("val/macro_f1", val_metrics["macro_f1"], epoch)

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_metrics['loss']:.4f} train_f1={train_metrics['macro_f1']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_f1={val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_id": args.model_id,
                    "state_dict": model.state_dict(),
                    "label_to_idx": label_to_idx,
                    "args": vars(args),
                },
                best_ckpt_path,
            )

    ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])

    test_eval = evaluate(model, test_loader, device)
    y_true = test_eval["y_true"]
    y_pred = test_eval["y_pred"]

    target_names = [idx_to_label[i] for i in range(len(idx_to_label))]
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(target_names))),
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(target_names))))

    _save_json(output_dir / "label_to_idx.json", label_to_idx)
    _save_json(output_dir / "split_stats.json", split_stats)
    _save_json(
        output_dir / "splits.json",
        {
            k: [
                {
                    "stem": s.stem,
                    "label": s.label_name,
                    "image": str(s.image_path),
                }
                for s in v
            ]
            for k, v in splits.items()
        },
    )
    _save_json(output_dir / "history.json", history)
    _save_json(output_dir / "classification_report_test.json", report)
    np.savetxt(output_dir / "confusion_matrix_test.csv", cm, fmt="%d", delimiter=",")

    summary = {
        "model_id": args.model_id,
        "device": device,
        "num_samples": len(samples),
        "num_classes": len(label_to_idx),
        "best_epoch": int(best_epoch),
        "best_val_macro_f1": float(best_val_f1),
        "test": {
            "loss": float(test_eval["loss"]),
            "accuracy": float(test_eval["accuracy"]),
            "macro_f1": float(test_eval["macro_f1"]),
        },
        "paths": {
            "best_checkpoint": str(best_ckpt_path.resolve()),
            "history": str((output_dir / "history.json").resolve()),
            "report": str((output_dir / "classification_report_test.json").resolve()),
            "confusion_matrix": str((output_dir / "confusion_matrix_test.csv").resolve()),
            "tensorboard_dir": str(tb_dir.resolve()) if writer is not None else "",
        },
    }
    _save_json(output_dir / "summary.json", summary)

    if writer is not None:
        writer.add_scalar("test/loss", test_eval["loss"], 0)
        writer.add_scalar("test/accuracy", test_eval["accuracy"], 0)
        writer.add_scalar("test/macro_f1", test_eval["macro_f1"], 0)
        writer.flush()
        writer.close()

    print("\nResumo final:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
