from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode, RandomResizedCrop
from torchvision.transforms import functional as TF


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS


def list_images_from_source(
    images_dir: str = "",
    list_txt: str = "",
    list_json: str = "",
    recursive: bool = False,
) -> list[Path]:
    if sum(bool(x) for x in (images_dir, list_txt, list_json)) != 1:
        raise ValueError("Use exatamente uma fonte: images_dir OU list_txt OU list_json")

    paths: list[Path] = []
    if images_dir:
        root = Path(images_dir)
        if not root.exists():
            raise FileNotFoundError(f"Pasta nao encontrada: {root}")
        it = root.rglob("*") if recursive else root.glob("*")
        paths = [p for p in it if _is_image_file(p)]
    elif list_txt:
        txt_path = Path(list_txt)
        if not txt_path.exists():
            raise FileNotFoundError(f"Arquivo txt nao encontrado: {txt_path}")
        for ln in txt_path.read_text(encoding="utf-8").splitlines():
            s = ln.strip()
            if not s or s.startswith("#"):
                continue
            p = Path(s)
            if p.exists() and _is_image_file(p):
                paths.append(p)
    else:
        json_path = Path(list_json)
        if not json_path.exists():
            raise FileNotFoundError(f"Arquivo json nao encontrado: {json_path}")
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        root = Path(payload.get("root_dir", ""))
        samples = payload.get("samples", [])
        for row in samples:
            rel = row.get("path", "")
            if not rel:
                continue
            p = (root / rel).resolve()
            if p.exists() and _is_image_file(p):
                paths.append(p)

    uniq = sorted(set(paths))
    if not uniq:
        raise RuntimeError("Nenhuma imagem valida encontrada na fonte informada")
    return uniq


def _pil_to_rgb(path: Path) -> Image.Image:
    img = Image.open(path)
    # Mantemos o dado em modo grayscale e replicamos para RGB para compatibilidade com DINOv2.
    gray = ImageOps.grayscale(img)
    return Image.merge("RGB", (gray, gray, gray))


@dataclass
class CropSpec:
    kind: str
    size: int
    scale: tuple[float, float]
    ratio: tuple[float, float]


class SSLMultiCropTransform:
    def __init__(
        self,
        global_crops: int = 2,
        local_crops: int = 6,
        global_size: int = 384,
        local_size: int = 192,
        global_scale: tuple[float, float] = (0.15, 1.0),
        local_scale: tuple[float, float] = (0.05, 0.15),
        ratio: tuple[float, float] = (0.9, 1.1),
        rotation_deg: float = 5.0,
        brightness_delta: float = 0.06,
        contrast_delta: float = 0.08,
        blur_prob: float = 0.1,
        noise_prob: float = 0.1,
        noise_std: float = 0.01,
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        self.global_crops = int(global_crops)
        self.local_crops = int(local_crops)
        self.rotation_deg = float(rotation_deg)
        self.brightness_delta = float(brightness_delta)
        self.contrast_delta = float(contrast_delta)
        self.blur_prob = float(blur_prob)
        self.noise_prob = float(noise_prob)
        self.noise_std = float(noise_std)
        self.mean = mean
        self.std = std
        self.specs: list[CropSpec] = []
        for _ in range(self.global_crops):
            self.specs.append(CropSpec("global", int(global_size), tuple(global_scale), tuple(ratio)))
        for _ in range(self.local_crops):
            self.specs.append(CropSpec("local", int(local_size), tuple(local_scale), tuple(ratio)))

    def _rand_crop(self, img: Image.Image, spec: CropSpec) -> tuple[Image.Image, dict[str, Any]]:
        w, h = img.size
        top, left, crop_h, crop_w = RandomResizedCrop.get_params(img, spec.scale, spec.ratio)
        cropped = TF.resized_crop(
            img,
            top,
            left,
            crop_h,
            crop_w,
            size=[spec.size, spec.size],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )
        area_scale = float((crop_h * crop_w) / max(1, w * h))
        ratio_hw = float(crop_w / max(1, crop_h))
        meta = {
            "kind": spec.kind,
            "size": int(spec.size),
            "top": int(top),
            "left": int(left),
            "crop_h": int(crop_h),
            "crop_w": int(crop_w),
            "src_h": int(h),
            "src_w": int(w),
            "area_scale": area_scale,
            "crop_ratio": ratio_hw,
        }
        return cropped, meta

    def _augment(self, img: Image.Image) -> Image.Image:
        out = img
        if self.rotation_deg > 0.0:
            ang = random.uniform(-self.rotation_deg, self.rotation_deg)
            out = TF.rotate(out, ang, interpolation=InterpolationMode.BILINEAR, fill=[0, 0, 0])

        if self.brightness_delta > 0.0:
            bf = random.uniform(1.0 - self.brightness_delta, 1.0 + self.brightness_delta)
            out = TF.adjust_brightness(out, bf)

        if self.contrast_delta > 0.0:
            cf = random.uniform(1.0 - self.contrast_delta, 1.0 + self.contrast_delta)
            out = TF.adjust_contrast(out, cf)

        if random.random() < self.blur_prob:
            out = TF.gaussian_blur(out, kernel_size=[3, 3], sigma=[0.1, 0.8])
        return out

    def _to_tensors(self, img: Image.Image) -> tuple[torch.Tensor, np.ndarray]:
        t = TF.to_tensor(img)
        if random.random() < self.noise_prob and self.noise_std > 0.0:
            t = torch.clamp(t + torch.randn_like(t) * self.noise_std, 0.0, 1.0)
        preview = (t.detach().cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
        n = TF.normalize(t, self.mean, self.std)
        return n, preview

    def sample_views(self, img: Image.Image) -> dict[str, Any]:
        student: list[torch.Tensor] = []
        teacher: list[torch.Tensor] = []
        previews: list[np.ndarray] = []
        metas: list[dict[str, Any]] = []
        for idx, spec in enumerate(self.specs):
            c, m = self._rand_crop(img, spec)
            c = self._augment(c)
            t, p = self._to_tensors(c)
            student.append(t)
            previews.append(p)
            metas.append(m)
            if idx < self.global_crops:
                teacher.append(t)
        return {
            "student_crops": student,
            "teacher_crops": teacher,
            "preview_images": previews,
            "meta": metas,
        }

    def __call__(self, img: Image.Image) -> dict[str, Any]:
        return self.sample_views(img)


class SSLImageDataset(Dataset):
    def __init__(self, paths: list[Path], transform: SSLMultiCropTransform) -> None:
        self.paths = list(paths)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        path = self.paths[idx]
        img = _pil_to_rgb(path)
        bundle = self.transform(img)
        return {
            "path": str(path),
            "student_crops": bundle["student_crops"],
            "teacher_crops": bundle["teacher_crops"],
            "preview_images": bundle["preview_images"],
            "meta": bundle["meta"],
        }


def collate_ssl_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        raise ValueError("Batch vazio")
    n_student = len(items[0]["student_crops"])
    n_teacher = len(items[0]["teacher_crops"])
    student_views = [torch.stack([it["student_crops"][i] for it in items], dim=0) for i in range(n_student)]
    teacher_views = [torch.stack([it["teacher_crops"][i] for it in items], dim=0) for i in range(n_teacher)]
    return {
        "paths": [it["path"] for it in items],
        "student_views": student_views,
        "teacher_views": teacher_views,
        "preview_images": [it["preview_images"] for it in items],
        "meta": [it["meta"] for it in items],
    }

