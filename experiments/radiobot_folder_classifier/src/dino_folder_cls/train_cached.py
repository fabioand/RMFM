from __future__ import annotations

import hashlib
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
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover
    SummaryWriter = None
from transformers import AutoImageProcessor

from .data import FolderDataset, build_label_mapping, discover_samples_from_list_json, make_collate_fn, stratified_split
from .model import FrozenDinoClassifier


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


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _class_distribution(samples, name: str) -> dict[str, int]:
    c = Counter(s.label_name for s in samples)
    print(f"{name}: {len(samples)} amostras | classes={len(c)}")
    return dict(sorted(c.items(), key=lambda kv: kv[0]))


def _split_signature(splits: dict[str, list[Any]]) -> str:
    payload: dict[str, list[dict[str, str]]] = {}
    for split_name in ("train", "val", "test"):
        items = []
        for s in splits[split_name]:
            items.append({"stem": s.stem, "label": s.label_name})
        payload[split_name] = sorted(items, key=lambda x: (x["stem"], x["label"]))
    raw = json.dumps(payload, ensure_ascii=True, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class HeadClassifier(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(dropout),
            nn.Linear(in_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@torch.no_grad()
def _extract_split_features(
    backbone: FrozenDinoClassifier,
    ds: FolderDataset,
    processor,
    batch_size: int,
    num_workers: int,
    shortest_edge: int,
    crop_size: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    collate_fn = make_collate_fn(processor, shortest_edge=shortest_edge, crop_size=crop_size)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=False,
    )

    feats_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []
    for batch in loader:
        pixel_values = batch["pixel_values"].to(device)
        labels = np.array(batch["labels"], dtype=np.int64)
        feats = backbone._extract_features(pixel_values).detach().cpu().numpy().astype(np.float32)
        feats_list.append(feats)
        labels_list.append(labels)

    x = np.concatenate(feats_list, axis=0) if feats_list else np.zeros((0, 1), dtype=np.float32)
    y = np.concatenate(labels_list, axis=0) if labels_list else np.zeros((0,), dtype=np.int64)
    return x, y


def _eval_head(model: HeadClassifier, loader: DataLoader, device: str) -> dict[str, Any]:
    model.eval()
    losses: list[float] = []
    y_true: list[int] = []
    y_pred: list[int] = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            pred = torch.argmax(logits, dim=1)
            losses.append(float(loss.item()))
            y_true.extend(yb.detach().cpu().tolist())
            y_pred.extend(pred.detach().cpu().tolist())

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "y_true": y_true,
        "y_pred": y_pred,
    }


def run_training_cached(args: Namespace) -> None:
    set_seed(int(args.seed))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    features_dir = output_dir / "features_cache"
    features_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = output_dir / "tb"

    list_json = Path(args.list_json)
    samples = discover_samples_from_list_json(list_json)
    if len(samples) == 0:
        raise SystemExit("Nenhuma amostra válida encontrada no JSON.")
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
    device = "cpu" if bool(args.cpu) else resolve_device()
    print(f"device={device}")

    ds_train = FolderDataset(splits["train"], label_to_idx)
    ds_val = FolderDataset(splits["val"], label_to_idx)
    ds_test = FolderDataset(splits["test"], label_to_idx)

    cache_meta_path = features_dir / "cache_meta.json"
    expected_meta = {
        "list_json": str(list_json.resolve()),
        "model_id": str(args.model_id),
        "shortest_edge": int(args.shortest_edge),
        "crop_size": int(args.crop_size),
        "seed": int(args.seed),
        "val_size": float(args.val_size),
        "test_size": float(args.test_size),
        "max_samples": int(args.max_samples),
        "label_to_idx": label_to_idx,
        "split_signature": _split_signature(splits),
    }
    feature_paths = {
        "x_train": features_dir / "x_train.npy",
        "y_train": features_dir / "y_train.npy",
        "x_val": features_dir / "x_val.npy",
        "y_val": features_dir / "y_val.npy",
        "x_test": features_dir / "x_test.npy",
        "y_test": features_dir / "y_test.npy",
    }
    cache_files_exist = all(p.exists() for p in feature_paths.values()) and cache_meta_path.exists()

    use_cached_features = False
    if cache_files_exist and not bool(args.force_reextract_features):
        try:
            cached_meta = _load_json(cache_meta_path)
            if cached_meta == expected_meta:
                use_cached_features = True
        except Exception:
            use_cached_features = False

    if use_cached_features:
        print("Reusando features cacheadas (setup idêntico).")
        x_train = np.load(feature_paths["x_train"]).astype(np.float32)
        y_train = np.load(feature_paths["y_train"]).astype(np.int64)
        x_val = np.load(feature_paths["x_val"]).astype(np.float32)
        y_val = np.load(feature_paths["y_val"]).astype(np.int64)
        x_test = np.load(feature_paths["x_test"]).astype(np.float32)
        y_test = np.load(feature_paths["y_test"]).astype(np.int64)
    else:
        print("Extraindo features (uma vez só)...")
        processor = AutoImageProcessor.from_pretrained(
            args.model_id,
            token=hf_token,
            local_files_only=bool(args.offline),
        )
        backbone_model = FrozenDinoClassifier(
            model_id=args.model_id,
            num_classes=len(label_to_idx),
            hf_token=hf_token,
            local_files_only=bool(args.offline),
            freeze_backbone=True,
            dropout=float(args.dropout),
        ).to(device)
        backbone_model.eval()

        x_train, y_train = _extract_split_features(
            backbone_model, ds_train, processor, int(args.feature_batch_size), int(args.num_workers),
            int(args.shortest_edge), int(args.crop_size), device
        )
        x_val, y_val = _extract_split_features(
            backbone_model, ds_val, processor, int(args.feature_batch_size), int(args.num_workers),
            int(args.shortest_edge), int(args.crop_size), device
        )
        x_test, y_test = _extract_split_features(
            backbone_model, ds_test, processor, int(args.feature_batch_size), int(args.num_workers),
            int(args.shortest_edge), int(args.crop_size), device
        )

        np.save(feature_paths["x_train"], x_train)
        np.save(feature_paths["y_train"], y_train)
        np.save(feature_paths["x_val"], x_val)
        np.save(feature_paths["y_val"], y_val)
        np.save(feature_paths["x_test"], x_test)
        np.save(feature_paths["y_test"], y_test)
        _save_json(cache_meta_path, expected_meta)

    in_dim = int(x_train.shape[1])
    head = HeadClassifier(in_dim=in_dim, num_classes=len(label_to_idx), dropout=float(args.dropout)).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val)),
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test)),
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=0,
    )

    writer = None
    if not bool(args.no_tensorboard) and SummaryWriter is not None:
        writer = SummaryWriter(log_dir=str(tb_dir))

    history: list[dict[str, Any]] = []
    best_val_f1 = -1.0
    best_epoch = -1
    best_ckpt_path = output_dir / "best_head_only.pt"

    for epoch in range(1, int(args.epochs) + 1):
        head.train()
        losses: list[float] = []
        y_true_train: list[int] = []
        y_pred_train: list[int] = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = head(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
            pred = torch.argmax(logits, dim=1)
            y_true_train.extend(yb.detach().cpu().tolist())
            y_pred_train.extend(pred.detach().cpu().tolist())

        train_metrics = {
            "loss": float(np.mean(losses)) if losses else 0.0,
            "accuracy": float(accuracy_score(y_true_train, y_pred_train)),
            "macro_f1": float(f1_score(y_true_train, y_pred_train, average="macro")),
        }
        val_metrics = _eval_head(head, val_loader, device)

        row = {"epoch": epoch, "train": train_metrics, "val": {k: val_metrics[k] for k in ("loss", "accuracy", "macro_f1")}}
        history.append(row)
        print(
            f"epoch={epoch:03d} train_loss={train_metrics['loss']:.4f} train_f1={train_metrics['macro_f1']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_f1={val_metrics['macro_f1']:.4f}"
        )

        if writer is not None:
            writer.add_scalar("train/loss", train_metrics["loss"], epoch)
            writer.add_scalar("train/accuracy", train_metrics["accuracy"], epoch)
            writer.add_scalar("train/macro_f1", train_metrics["macro_f1"], epoch)
            writer.add_scalar("val/loss", val_metrics["loss"], epoch)
            writer.add_scalar("val/accuracy", val_metrics["accuracy"], epoch)
            writer.add_scalar("val/macro_f1", val_metrics["macro_f1"], epoch)

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_id": args.model_id,
                    "head_state_dict": head.state_dict(),
                    "label_to_idx": label_to_idx,
                    "in_dim": in_dim,
                    "args": vars(args),
                },
                best_ckpt_path,
            )

    ckpt = torch.load(best_ckpt_path, map_location=device)
    head.load_state_dict(ckpt["head_state_dict"])
    test_eval = _eval_head(head, test_loader, device)

    target_names = [idx_to_label[i] for i in range(len(idx_to_label))]
    report = classification_report(
        test_eval["y_true"],
        test_eval["y_pred"],
        labels=list(range(len(target_names))),
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(test_eval["y_true"], test_eval["y_pred"], labels=list(range(len(target_names))))

    _save_json(output_dir / "label_to_idx.json", label_to_idx)
    _save_json(output_dir / "split_stats.json", split_stats)
    _save_json(output_dir / "history.json", history)
    _save_json(output_dir / "classification_report_test.json", report)
    np.savetxt(output_dir / "confusion_matrix_test.csv", cm, fmt="%d", delimiter=",")

    summary = {
        "mode": "cached_features_head_only",
        "list_json": str(list_json.resolve()),
        "model_id": args.model_id,
        "device": device,
        "num_samples": len(samples),
        "num_classes": len(label_to_idx),
        "feature_dim": in_dim,
        "features_cache_reused": bool(use_cached_features),
        "force_reextract_features": bool(args.force_reextract_features),
        "best_epoch": int(best_epoch),
        "best_val_macro_f1": float(best_val_f1),
        "test": {
            "loss": float(test_eval["loss"]),
            "accuracy": float(test_eval["accuracy"]),
            "macro_f1": float(test_eval["macro_f1"]),
        },
        "paths": {
            "best_checkpoint": str(best_ckpt_path.resolve()),
            "features_dir": str(features_dir.resolve()),
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
