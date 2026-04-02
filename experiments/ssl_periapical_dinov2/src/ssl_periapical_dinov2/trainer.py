from __future__ import annotations

import json
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from .config import save_json
from .data import SSLImageDataset, SSLMultiCropTransform, collate_ssl_batch, list_images_from_source
from .ssl_core import (
    build_ssl_bundle,
    cosine_lr,
    cosine_momentum,
    extract_feature,
    update_teacher_ema,
)
from .visuals import capture_ssl_epoch_visuals

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover
    SummaryWriter = None


@dataclass
class EpochRow:
    epoch: int
    train_loss: float
    lr: float
    momentum: float
    samples: int
    seconds: float
    throughput_img_s: float


def resolve_device(force_cpu: bool = False) -> str:
    if force_cpu:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_token(explicit: str = "", env_key: str = "HF_TOKEN") -> str | None:
    if explicit:
        return explicit
    return os.environ.get(env_key)


def _build_transform(cfg: dict[str, Any], processor) -> SSLMultiCropTransform:
    ds = cfg["dataset"]
    mc = cfg["multicrop"]
    aug = cfg["augmentation"]
    mean = tuple(float(x) for x in getattr(processor, "image_mean", [0.485, 0.456, 0.406]))
    std = tuple(float(x) for x in getattr(processor, "image_std", [0.229, 0.224, 0.225]))
    return SSLMultiCropTransform(
        global_crops=int(mc["global_crops"]),
        local_crops=int(mc["local_crops"]),
        global_size=int(mc["global_size"]),
        local_size=int(mc["local_size"]),
        global_scale=tuple(float(x) for x in mc["global_scale"]),
        local_scale=tuple(float(x) for x in mc["local_scale"]),
        ratio=tuple(float(x) for x in mc["ratio"]),
        rotation_deg=float(aug["rotation_deg"]),
        brightness_delta=float(aug["brightness_delta"]),
        contrast_delta=float(aug["contrast_delta"]),
        blur_prob=float(aug["blur_prob"]),
        noise_prob=float(aug["noise_prob"]),
        noise_std=float(aug["noise_std"]),
        mean=mean,
        std=std,
    )


def _write_config_resolved(run_dir: Path, cfg: dict[str, Any]) -> None:
    out = run_dir / "config_resolved.json"
    out.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")


def _extract_block_idx(name: str) -> int | None:
    m = re.search(r"(?:encoder\.layer|blocks)\.(\d+)\.", name)
    if not m:
        return None
    return int(m.group(1))


def _apply_backbone_trainable_mask(
    backbone: torch.nn.Module,
    train_from_block: int,
    freeze_patch_embed: bool,
) -> tuple[int, int]:
    total = 0
    trainable = 0
    for name, p in backbone.named_parameters():
        total += p.numel()
        keep = True
        block_idx = _extract_block_idx(name)
        if block_idx is not None and block_idx < train_from_block:
            keep = False
        if freeze_patch_embed and ("embeddings" in name or "patch_embed" in name or "patch_embeddings" in name):
            keep = False
        p.requires_grad = keep
        if keep:
            trainable += p.numel()
    return trainable, total


def _resolve_train_from_block(cfg: dict[str, Any], epoch: int, num_blocks: int) -> int:
    tr = cfg["training"]
    schedule = tr.get("unfreeze_schedule", [])
    if not schedule:
        return int(tr.get("train_from_block", 0))
    start = int(schedule[0].get("train_from_block", 0))
    for row in schedule:
        e = int(row.get("epoch", 1))
        if epoch >= e:
            start = int(row.get("train_from_block", start))
    return max(0, min(start, max(0, num_blocks - 1)))


def _resolve_snapshot_epochs(cfg: dict[str, Any], epochs: int) -> set[int]:
    tr = cfg["training"]
    xs = tr.get("snapshot_epochs", [])
    out: set[int] = set()
    for x in xs:
        try:
            e = int(x)
        except Exception:
            continue
        if 1 <= e <= epochs:
            out.add(e)
    return out


def _load_eval_tensor(path: Path, size: int, mean: tuple[float, ...], std: tuple[float, ...], device: str) -> torch.Tensor:
    img = Image.open(path)
    gray = ImageOps.grayscale(img)
    rgb = Image.merge("RGB", (gray, gray, gray))
    rgb = TF.resize(rgb, [size, size], interpolation=InterpolationMode.BICUBIC, antialias=True)
    t = TF.to_tensor(rgb)
    t = TF.normalize(t, mean, std)
    return t.unsqueeze(0).to(device)


@torch.no_grad()
def _compute_collapse_diagnostics(
    sample_paths: list[Path],
    backbone: torch.nn.Module,
    device: str,
    image_size: int,
    mean: tuple[float, ...],
    std: tuple[float, ...],
    batch_size: int,
) -> dict[str, Any]:
    feats: list[np.ndarray] = []
    batch_tensors: list[torch.Tensor] = []
    valid = 0
    for p in sample_paths:
        try:
            batch_tensors.append(_load_eval_tensor(p, image_size, mean, std, device))
            valid += 1
            if len(batch_tensors) >= batch_size:
                xb = torch.cat(batch_tensors, dim=0)
                out = backbone(pixel_values=xb)
                feats.append(extract_feature(out).detach().cpu().numpy().astype(np.float32))
                batch_tensors = []
        except Exception:
            continue
    if batch_tensors:
        xb = torch.cat(batch_tensors, dim=0)
        out = backbone(pixel_values=xb)
        feats.append(extract_feature(out).detach().cpu().numpy().astype(np.float32))
    if not feats:
        return {
            "num_samples": 0,
            "feature_dim": 0,
            "pc1_explained_variance_ratio": 0.0,
            "top5_explained_variance_ratio": 0.0,
            "embedding_norm_mean": 0.0,
            "embedding_norm_std": 0.0,
        }
    x = np.concatenate(feats, axis=0)
    x = x - x.mean(axis=0, keepdims=True)
    # SVD no eixo das features para medir anisotropia.
    _, s, _ = np.linalg.svd(x, full_matrices=False)
    eig = np.square(s)
    eig_sum = float(np.sum(eig)) + 1e-12
    pc1 = float(eig[0] / eig_sum) if eig.size else 0.0
    top5 = float(np.sum(eig[: min(5, eig.size)]) / eig_sum) if eig.size else 0.0
    norms = np.linalg.norm(x, axis=1)
    return {
        "num_samples": int(valid),
        "feature_dim": int(x.shape[1]),
        "pc1_explained_variance_ratio": pc1,
        "top5_explained_variance_ratio": top5,
        "embedding_norm_mean": float(norms.mean()),
        "embedding_norm_std": float(norms.std()),
    }


def run_ssl_training(cfg: dict[str, Any]) -> Path:
    seed = int(cfg["training"]["seed"])
    set_seed(seed)

    run_name = str(cfg["run"]["name"])
    out_base = Path(cfg["run"]["output_dir"])
    run_dir = out_base / run_name
    ckpt_dir = run_dir / "checkpoints"
    tb_dir = run_dir / "tb"
    visual_dir = run_dir / "train_visuals"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    visual_dir.mkdir(parents=True, exist_ok=True)
    _write_config_resolved(run_dir, cfg)

    device = resolve_device(force_cpu=bool(cfg["training"].get("cpu", False)))
    token = _resolve_token(str(cfg["model"].get("hf_token", "")), str(cfg["model"].get("hf_token_env", "HF_TOKEN")))
    model_id = str(cfg["model"]["model_id"])
    local_only = bool(cfg["model"].get("offline", False))

    processor = AutoImageProcessor.from_pretrained(model_id, token=token, local_files_only=local_only)
    transform = _build_transform(cfg, processor)
    mean = tuple(float(x) for x in getattr(processor, "image_mean", [0.485, 0.456, 0.406]))
    std = tuple(float(x) for x in getattr(processor, "image_std", [0.229, 0.224, 0.225]))

    paths = list_images_from_source(
        images_dir=str(cfg["dataset"].get("images_dir", "")),
        list_txt=str(cfg["dataset"].get("list_txt", "")),
        list_json=str(cfg["dataset"].get("list_json", "")),
        recursive=bool(cfg["dataset"].get("recursive", False)),
    )
    max_images = int(cfg["dataset"].get("max_images", 0))
    if max_images > 0:
        paths = paths[:max_images]

    dataset = SSLImageDataset(paths, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["training"]["num_workers"]),
        pin_memory=False,
        collate_fn=collate_ssl_batch,
        drop_last=True,
    )

    bundle = build_ssl_bundle(
        model_id=model_id,
        out_dim=int(cfg["model"]["out_dim"]),
        device=device,
        local_files_only=local_only,
        hf_token=token,
        head_hidden_dim=int(cfg["model"]["head_hidden_dim"]),
        head_bottleneck_dim=int(cfg["model"]["head_bottleneck_dim"]),
        student_temp=float(cfg["loss"]["student_temp"]),
        center_momentum=float(cfg["loss"]["center_momentum"]),
        teacher_temp_warmup=float(cfg["loss"]["teacher_temp_warmup"]),
        teacher_temp=float(cfg["loss"]["teacher_temp"]),
        teacher_temp_warmup_epochs=int(cfg["loss"]["teacher_temp_warmup_epochs"]),
        total_epochs=int(cfg["training"]["epochs"]),
    )

    for p in bundle.student_head.parameters():
        p.requires_grad = True
    tr = cfg["training"]
    lr_fallback = float(tr.get("lr", 1e-4))
    lr_backbone_base = float(tr.get("lr_backbone", lr_fallback))
    lr_head_base = float(tr.get("lr_head", lr_fallback))
    min_lr_fallback = float(tr.get("min_lr", 1e-6))
    min_lr_backbone = float(tr.get("min_lr_backbone", min_lr_fallback))
    min_lr_head = float(tr.get("min_lr_head", min_lr_fallback))
    freeze_patch_embed = bool(tr.get("freeze_patch_embed", False))
    num_blocks = int(getattr(getattr(bundle.student_backbone, "config", object()), "num_hidden_layers", 12))

    optimizer = torch.optim.AdamW(
        [
            {
                "name": "backbone",
                "params": list(bundle.student_backbone.parameters()),
                "lr": lr_backbone_base,
            },
            {
                "name": "head",
                "params": list(bundle.student_head.parameters()),
                "lr": lr_head_base,
            },
        ],
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg["training"].get("amp", True) and device == "cuda"))

    epochs = int(cfg["training"]["epochs"])
    steps_per_epoch = len(loader)
    if steps_per_epoch < 1:
        raise RuntimeError("DataLoader sem batches. Ajuste batch_size/max_images.")
    total_steps = steps_per_epoch * epochs
    warmup_steps = int(cfg["training"].get("warmup_steps", 0))
    snapshot_epochs = _resolve_snapshot_epochs(cfg, epochs)

    writer = None
    if SummaryWriter is not None and not bool(cfg["training"].get("no_tensorboard", False)):
        writer = SummaryWriter(log_dir=str(tb_dir))

    history: list[dict[str, Any]] = []
    global_step = 0
    best_loss = float("inf")
    best_ckpt = ckpt_dir / "best.pt"
    last_ckpt = ckpt_dir / "last.pt"
    collapse_dir = run_dir / "collapse_diagnostics"
    collapse_dir.mkdir(parents=True, exist_ok=True)
    snapshot_manifest_path = ckpt_dir / "snapshot_manifest.json"
    snapshot_manifest: list[dict[str, Any]] = []

    fixed_visual_paths = paths[: int(cfg["visuals"]["max_samples"])]
    collapse_cfg = cfg.get("collapse", {})
    collapse_enabled = bool(collapse_cfg.get("enabled", True))
    collapse_interval = int(collapse_cfg.get("interval_epochs", 10))
    collapse_max_samples = int(collapse_cfg.get("max_samples", 256))
    collapse_batch_size = int(collapse_cfg.get("batch_size", 32))
    collapse_image_size = int(collapse_cfg.get("image_size", int(cfg["multicrop"]["global_size"])))
    collapse_backbone_key = str(collapse_cfg.get("backbone_key", "teacher"))
    collapse_sample_paths = paths[: max(0, collapse_max_samples)]
    log_every_steps = int(cfg["training"].get("log_every_steps", 10))

    if bool(cfg["visuals"].get("capture_on_start", True)):
        capture_ssl_epoch_visuals(
            out_dir=visual_dir,
            epoch=0,
            sample_paths=fixed_visual_paths,
            transform=transform,
            student_backbone=bundle.student_backbone,
            device=device,
            max_samples=int(cfg["visuals"]["max_samples"]),
            interval=1,
        )
        print("[VISUAL] captura inicial salva em epoch_0000")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_from_block = _resolve_train_from_block(cfg, epoch, num_blocks)
        bb_trainable, bb_total = _apply_backbone_trainable_mask(
            bundle.student_backbone,
            train_from_block=train_from_block,
            freeze_patch_embed=freeze_patch_embed,
        )
        bundle.student_backbone.train()
        bundle.student_head.train()
        if bb_trainable < bb_total:
            # Mantemos BN/LN em modo train no bloco ativo; o grad define o que atualiza.
            pass

        losses: list[float] = []
        samples_seen = 0
        last_lr = lr_backbone_base
        last_lr_head = lr_head_base
        last_m = float(cfg["training"]["teacher_momentum_base"])

        for batch_idx, batch in enumerate(loader, start=1):
            last_lr = cosine_lr(
                global_step,
                total_steps=total_steps,
                base_lr=lr_backbone_base,
                min_lr=min_lr_backbone,
                warmup_steps=warmup_steps,
            )
            last_lr_head = cosine_lr(
                global_step,
                total_steps=total_steps,
                base_lr=lr_head_base,
                min_lr=min_lr_head,
                warmup_steps=warmup_steps,
            )
            for g in optimizer.param_groups:
                gname = str(g.get("name", ""))
                if gname == "head":
                    g["lr"] = last_lr_head
                else:
                    g["lr"] = last_lr

            student_views = [v.to(device, non_blocking=True) for v in batch["student_views"]]
            teacher_views = [v.to(device, non_blocking=True) for v in batch["teacher_views"]]
            bsz = int(student_views[0].shape[0])
            samples_seen += bsz

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                s_out: list[torch.Tensor] = []
                for v in student_views:
                    out = bundle.student_backbone(pixel_values=v)
                    feat = extract_feature(out)
                    s_out.append(bundle.student_head(feat))

                with torch.no_grad():
                    t_out: list[torch.Tensor] = []
                    for v in teacher_views:
                        out = bundle.teacher_backbone(pixel_values=v)
                        feat = extract_feature(out)
                        t_out.append(bundle.teacher_head(feat))

                loss = bundle.dino_loss(s_out, t_out, epoch=epoch)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                if float(cfg["training"].get("grad_clip_norm", 0.0)) > 0.0:
                    scaler.unscale_(optimizer)
                    active_trainable = [p for p in bundle.student_backbone.parameters() if p.requires_grad] + list(
                        bundle.student_head.parameters()
                    )
                    torch.nn.utils.clip_grad_norm_(active_trainable, float(cfg["training"]["grad_clip_norm"]))
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if float(cfg["training"].get("grad_clip_norm", 0.0)) > 0.0:
                    active_trainable = [p for p in bundle.student_backbone.parameters() if p.requires_grad] + list(
                        bundle.student_head.parameters()
                    )
                    torch.nn.utils.clip_grad_norm_(active_trainable, float(cfg["training"]["grad_clip_norm"]))
                optimizer.step()

            last_m = cosine_momentum(
                global_step,
                total_steps=total_steps,
                base_m=float(cfg["training"]["teacher_momentum_base"]),
                final_m=float(cfg["training"]["teacher_momentum_final"]),
            )
            update_teacher_ema(
                student_backbone=bundle.student_backbone,
                teacher_backbone=bundle.teacher_backbone,
                student_head=bundle.student_head,
                teacher_head=bundle.teacher_head,
                momentum=last_m,
            )

            lv = float(loss.detach().item())
            losses.append(lv)
            if writer is not None:
                writer.add_scalar("train/loss_step", lv, global_step)
                writer.add_scalar("train/lr_backbone", last_lr, global_step)
                writer.add_scalar("train/lr_head", last_lr_head, global_step)
                writer.add_scalar("train/teacher_momentum", last_m, global_step)

            if (
                epoch == 1
                and log_every_steps > 0
                and (batch_idx % log_every_steps == 0 or batch_idx == 1 or batch_idx == steps_per_epoch)
            ):
                elapsed_step = max(1e-6, time.time() - t0)
                thr_step = float(samples_seen / elapsed_step)
                print(
                    f"epoch={epoch:03d} batch={batch_idx:04d}/{steps_per_epoch:04d} "
                    f"loss={lv:.4f} lr_bb={last_lr:.6g} lr_head={last_lr_head:.6g} m={last_m:.6f} "
                    f"seen={samples_seen} throughput={thr_step:.2f} img/s"
                )
            global_step += 1

        elapsed = max(1e-6, time.time() - t0)
        mean_loss = float(np.mean(losses)) if losses else 0.0
        thr = float(samples_seen / elapsed)
        row = EpochRow(
            epoch=epoch,
            train_loss=mean_loss,
            lr=last_lr,
            momentum=last_m,
            samples=samples_seen,
            seconds=elapsed,
            throughput_img_s=thr,
        )
        history.append(row.__dict__)

        if writer is not None:
            writer.add_scalar("train/loss_epoch", row.train_loss, epoch)
            writer.add_scalar("train/throughput_img_s", row.throughput_img_s, epoch)
            writer.add_scalar("train/backbone_trainable_ratio", float(bb_trainable / max(1, bb_total)), epoch)
            writer.add_scalar("train/train_from_block", float(train_from_block), epoch)

        head_trainable = sum(p.numel() for p in bundle.student_head.parameters() if p.requires_grad)
        head_total = sum(p.numel() for p in bundle.student_head.parameters())

        ckpt_payload = {
            "epoch": epoch,
            "global_step": global_step,
            "config": cfg,
            "student_backbone": bundle.student_backbone.state_dict(),
            "teacher_backbone": bundle.teacher_backbone.state_dict(),
            "student_head": bundle.student_head.state_dict(),
            "teacher_head": bundle.teacher_head.state_dict(),
            "dino_loss": bundle.dino_loss.state_dict(),
            "optimizer": optimizer.state_dict(),
            "history": history,
            "model_id": model_id,
            "freeze_state": {
                "train_from_block": int(train_from_block),
                "freeze_patch_embed": bool(freeze_patch_embed),
                "backbone_trainable_params": int(bb_trainable),
                "backbone_total_params": int(bb_total),
                "head_trainable_params": int(head_trainable),
                "head_total_params": int(head_total),
            },
        }
        torch.save(ckpt_payload, last_ckpt)
        if mean_loss < best_loss:
            best_loss = mean_loss
            torch.save(ckpt_payload, best_ckpt)
        if epoch in snapshot_epochs:
            snapshot_path = ckpt_dir / f"epoch_{epoch:03d}_snapshot.pt"
            torch.save(ckpt_payload, snapshot_path)
            snapshot_manifest.append(
                {
                    "epoch": int(epoch),
                    "path": str(snapshot_path.resolve()),
                    "train_loss": float(mean_loss),
                    "lr_backbone": float(last_lr),
                    "lr_head": float(last_lr_head),
                }
            )

        capture_ssl_epoch_visuals(
            out_dir=visual_dir,
            epoch=epoch,
            sample_paths=fixed_visual_paths,
            transform=transform,
            student_backbone=bundle.student_backbone,
            device=device,
            max_samples=int(cfg["visuals"]["max_samples"]),
            interval=int(cfg["visuals"]["interval"]),
        )

        if collapse_enabled and collapse_interval > 0 and (epoch % collapse_interval == 0):
            backbone_for_diag = bundle.teacher_backbone if collapse_backbone_key == "teacher" else bundle.student_backbone
            was_train_diag = backbone_for_diag.training
            backbone_for_diag.eval()
            diag = _compute_collapse_diagnostics(
                sample_paths=collapse_sample_paths,
                backbone=backbone_for_diag,
                device=device,
                image_size=collapse_image_size,
                mean=mean,
                std=std,
                batch_size=collapse_batch_size,
            )
            diag.update(
                {
                    "epoch": int(epoch),
                    "backbone_key": "teacher_backbone" if collapse_backbone_key == "teacher" else "student_backbone",
                }
            )
            save_json(collapse_dir / f"epoch_{epoch:03d}.json", diag)
            if writer is not None:
                writer.add_scalar("collapse/pc1_ratio", float(diag["pc1_explained_variance_ratio"]), epoch)
                writer.add_scalar("collapse/top5_ratio", float(diag["top5_explained_variance_ratio"]), epoch)
                writer.add_scalar("collapse/emb_norm_mean", float(diag["embedding_norm_mean"]), epoch)
            if was_train_diag:
                backbone_for_diag.train()

        print(
            f"epoch={epoch:03d} loss={mean_loss:.4f} lr_bb={last_lr:.6g} lr_head={last_lr_head:.6g} "
            f"m={last_m:.6f} thr={thr:.2f} img/s train_from_block={train_from_block} "
            f"bb_trainable={bb_trainable}/{bb_total}"
        )

    save_json(snapshot_manifest_path, {"snapshots": snapshot_manifest})

    summary = {
        "mode": "ssl_dinov2_teacher_student_multicrop",
        "model_id": model_id,
        "device": device,
        "num_images": len(paths),
        "epochs": epochs,
        "steps_per_epoch": steps_per_epoch,
        "best_train_loss": best_loss,
        "optimizer_groups": {
            "lr_backbone_base": lr_backbone_base,
            "lr_head_base": lr_head_base,
            "min_lr_backbone": min_lr_backbone,
            "min_lr_head": min_lr_head,
        },
        "paths": {
            "run_dir": str(run_dir.resolve()),
            "best_checkpoint": str(best_ckpt.resolve()),
            "last_checkpoint": str(last_ckpt.resolve()),
            "snapshot_manifest": str(snapshot_manifest_path.resolve()),
            "history": str((run_dir / "history.json").resolve()),
            "summary": str((run_dir / "summary.json").resolve()),
            "tensorboard_dir": str(tb_dir.resolve()) if writer is not None else "",
            "visuals_dir": str(visual_dir.resolve()),
            "collapse_diagnostics_dir": str(collapse_dir.resolve()),
        },
    }
    save_json(run_dir / "history.json", history)
    save_json(run_dir / "summary.json", summary)

    if writer is not None:
        writer.flush()
        writer.close()

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return run_dir
