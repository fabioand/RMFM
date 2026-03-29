from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class Sample:
    image_path: Path
    stem: str
    label_name: str


def _read_label_name(json_path: Path) -> str | None:
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    labels = payload.get("labels", [])
    if not isinstance(labels, list) or len(labels) == 0:
        return None

    # Dataset atual usa, em geral, uma única classe por arquivo.
    return str(labels[0])


def discover_samples(images_dir: Path, labels_dir: Path) -> list[Sample]:
    images_by_stem: dict[str, Path] = {}
    for p in images_dir.iterdir():
        if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTS:
            images_by_stem[p.stem] = p

    samples: list[Sample] = []
    for j in sorted(labels_dir.glob("*.json")):
        stem = j.stem
        image_path = images_by_stem.get(stem)
        if image_path is None:
            continue

        label_name = _read_label_name(j)
        if label_name is None:
            continue

        samples.append(Sample(image_path=image_path, stem=stem, label_name=label_name))

    return samples


def build_label_mapping(samples: list[Sample]) -> dict[str, int]:
    labels = sorted({s.label_name for s in samples})
    return {name: idx for idx, name in enumerate(labels)}


def stratified_split(
    samples: list[Sample],
    val_size: float,
    test_size: float,
    seed: int,
) -> dict[str, list[Sample]]:
    if not 0.0 < val_size < 1.0:
        raise ValueError("val_size deve estar em (0,1)")
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size deve estar em (0,1)")
    if val_size + test_size >= 1.0:
        raise ValueError("val_size + test_size deve ser < 1")

    y = [s.label_name for s in samples]
    idx = list(range(len(samples)))

    can_stratify = all(y.count(c) >= 3 for c in set(y))

    if can_stratify:
        try:
            train_idx, temp_idx = train_test_split(
                idx,
                test_size=(val_size + test_size),
                random_state=seed,
                stratify=y,
            )
        except ValueError:
            train_idx, temp_idx = train_test_split(
                idx,
                test_size=(val_size + test_size),
                random_state=seed,
                shuffle=True,
            )
        y_temp = [y[i] for i in temp_idx]
        val_ratio_within_temp = val_size / (val_size + test_size)
        try:
            val_idx, test_idx = train_test_split(
                temp_idx,
                test_size=(1.0 - val_ratio_within_temp),
                random_state=seed,
                stratify=y_temp,
            )
        except ValueError:
            val_idx, test_idx = train_test_split(
                temp_idx,
                test_size=(1.0 - val_ratio_within_temp),
                random_state=seed,
                shuffle=True,
            )
    else:
        train_idx, temp_idx = train_test_split(
            idx,
            test_size=(val_size + test_size),
            random_state=seed,
            shuffle=True,
        )
        val_ratio_within_temp = val_size / (val_size + test_size)
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=(1.0 - val_ratio_within_temp),
            random_state=seed,
            shuffle=True,
        )

    def _pick(indices: list[int]) -> list[Sample]:
        return [samples[i] for i in indices]

    return {
        "train": _pick(sorted(train_idx)),
        "val": _pick(sorted(val_idx)),
        "test": _pick(sorted(test_idx)),
    }


class PeriapicalDataset(Dataset):
    def __init__(self, samples: list[Sample], label_to_idx: dict[str, int]) -> None:
        self.samples = samples
        self.label_to_idx = label_to_idx

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        s = self.samples[index]
        img = Image.open(s.image_path).convert("RGB")
        return {
            "image": img,
            "label": self.label_to_idx[s.label_name],
            "label_name": s.label_name,
            "path": str(s.image_path),
            "stem": s.stem,
        }


def make_collate_fn(processor, shortest_edge: int = 0, crop_size: int = 0):
    processor_kwargs: dict[str, Any] = {}
    if shortest_edge > 0:
        processor_kwargs["size"] = {"shortest_edge": int(shortest_edge)}
    if crop_size > 0:
        processor_kwargs["crop_size"] = {
            "height": int(crop_size),
            "width": int(crop_size),
        }

    def _collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        images = [b["image"] for b in batch]
        labels = [int(b["label"]) for b in batch]
        paths = [b["path"] for b in batch]
        stems = [b["stem"] for b in batch]

        inputs = processor(images=images, return_tensors="pt", **processor_kwargs)
        return {
            "pixel_values": inputs["pixel_values"],
            "labels": labels,
            "paths": paths,
            "stems": stems,
        }

    return _collate
