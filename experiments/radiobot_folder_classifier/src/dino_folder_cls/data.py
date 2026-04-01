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


def _resolve_path(raw_path: str, root_dir: Path | None, list_json_dir: Path) -> Path:
    p = Path(raw_path)
    if p.is_absolute():
        return p
    if root_dir is not None:
        return (root_dir / p).resolve()
    return (list_json_dir / p).resolve()


def discover_samples_from_list_json(list_json_path: Path) -> list[Sample]:
    payload = json.loads(list_json_path.read_text(encoding="utf-8"))
    root_dir: Path | None = None
    list_json_dir = list_json_path.resolve().parent

    if isinstance(payload, dict) and payload.get("root_dir"):
        root_dir = Path(str(payload["root_dir"])).expanduser().resolve()

    if isinstance(payload, dict) and isinstance(payload.get("samples"), list):
        raw_items = payload["samples"]
    elif isinstance(payload, list):
        raw_items = payload
    else:
        raise ValueError("JSON de lista inválido: esperado dict com 'samples' ou list.")

    samples: list[Sample] = []
    for item in raw_items:
        if isinstance(item, dict):
            raw_path = item.get("path") or item.get("image") or item.get("image_path")
            folder = item.get("folder")
        elif isinstance(item, str):
            raw_path = item
            folder = None
        else:
            continue

        if not raw_path:
            continue
        p = _resolve_path(str(raw_path), root_dir=root_dir, list_json_dir=list_json_dir)
        if not p.exists() or not p.is_file() or p.suffix.lower() not in VALID_IMAGE_EXTS:
            continue
        if p.name.startswith("._") or "__MACOSX" in p.parts:
            continue

        label_name = str(folder) if folder else p.parent.name
        samples.append(Sample(image_path=p, stem=p.stem, label_name=label_name))

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


class FolderDataset(Dataset):
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
