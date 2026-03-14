from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .caa import ClassAwareAugmentor
from .types import ADItem


def _is_image_file(path: Path) -> bool:
    return path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _list_images(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    return sorted([p for p in folder.rglob("*") if p.is_file() and _is_image_file(p)])


def _find_mask(gt_dir: Path, img_path: Path) -> str:
    """
    Try common ground-truth naming patterns:
      - same filename
      - <stem>_mask.<ext>
    Also supports mismatched image/mask extensions (e.g., image .bmp but mask .png).
    Returns relative path to dataset root as posix, or "" if not found.
    """
    cand = gt_dir / img_path.name
    if cand.is_file():
        return cand

    exts = [
        img_path.suffix,
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".tif",
        ".tiff",
        ".webp",
    ]
    seen = set()
    for ext in exts:
        ext = (ext or "").lower()
        if not ext:
            continue
        if not ext.startswith("."):
            ext = "." + ext
        if ext in seen:
            continue
        seen.add(ext)

        cand = gt_dir / f"{img_path.stem}{ext}"
        if cand.is_file():
            return cand
        cand = gt_dir / f"{img_path.stem}_mask{ext}"
        if cand.is_file():
            return cand
    return ""


class MVTecStyleDataset(Dataset):
    """
    Parses a MVTec-style directory layout:
      root/<cls>/train/<good_name>/*
      root/<cls>/test/<type>/*
      root/<cls>/<gt_dir_name>/<type>/*_mask.png

    Also supports a "single class" root that directly contains train/test.
    """

    def __init__(
        self,
        root: str,
        split: str,
        image_transform: Callable[[Image.Image], torch.Tensor],
        mask_transform: Callable[[Image.Image], torch.Tensor],
        good_name: str = "good",
        gt_dir_name: str = "ground_truth",
        caa: Optional[ClassAwareAugmentor] = None,
        class_names: Optional[Sequence[str]] = None,
    ) -> None:
        self.root = os.fspath(root)
        self.split = split
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.good_name = good_name
        self.gt_dir_name = gt_dir_name
        self.caa = caa

        root_path = Path(self.root)
        if not root_path.exists():
            raise FileNotFoundError(self.root)

        categories = self._discover_categories(root_path, class_names=class_names)
        self.class_names = sorted([c.name for c in categories])
        self.class_to_id = {name: i for i, name in enumerate(self.class_names)}

        self.items: List[ADItem] = []
        for cat in categories:
            self.items.extend(self._build_items_for_category(root_path, cat))

        self._indices_by_class: Dict[str, List[int]] = {c: [] for c in self.class_names}
        self._indices_by_class_and_label: Dict[Tuple[str, int], List[int]] = {}
        for idx, item in enumerate(self.items):
            self._indices_by_class[item.cls_name].append(idx)
            self._indices_by_class_and_label.setdefault((item.cls_name, item.anomaly), []).append(idx)

        if self.caa is not None:
            self.caa.bind(self)

    def _discover_categories(self, root_path: Path, class_names: Optional[Sequence[str]] = None) -> List[Path]:
        if (root_path / "train").is_dir() and (root_path / "test").is_dir():
            return [root_path]

        candidates = [p for p in root_path.iterdir() if p.is_dir() and (p / "train").is_dir() and (p / "test").is_dir()]
        if class_names is None:
            return sorted(candidates)

        name_set = {str(x) for x in class_names}
        filtered = [p for p in candidates if p.name in name_set]
        if not filtered:
            raise ValueError(f"None of requested class_names exist under {root_path}: {sorted(name_set)}")
        return sorted(filtered)

    def _build_items_for_category(self, root_path: Path, category_path: Path) -> List[ADItem]:
        cls_name = category_path.name if category_path != root_path else root_path.name
        items: List[ADItem] = []

        split_dir = category_path / self.split
        if not split_dir.is_dir():
            # allow split alias: some datasets use "val" etc, user can pass --split accordingly
            raise FileNotFoundError(f"Split folder not found: {split_dir}")

        gt_root = category_path / self.gt_dir_name

        for defect_dir in sorted([p for p in split_dir.iterdir() if p.is_dir()]):
            specie = defect_dir.name
            anomaly = 0 if specie == self.good_name else 1
            for img_path in _list_images(defect_dir):
                rel_img = img_path.relative_to(root_path).as_posix()
                mask_rel = ""
                if anomaly == 1:
                    gt_dir = gt_root / specie
                    mask_path = _find_mask(gt_dir, img_path)
                    if mask_path:
                        mask_rel = Path(mask_path).relative_to(root_path).as_posix()
                items.append(
                    ADItem(
                        img_path=rel_img,
                        mask_path=mask_rel,
                        cls_name=cls_name,
                        specie_name=specie,
                        anomaly=anomaly,
                    )
                )

        return items

    def __len__(self) -> int:
        return len(self.items)

    def load_raw(self, index: int) -> Tuple[Image.Image, Image.Image, ADItem]:
        item = self.items[index]
        img = Image.open(os.path.join(self.root, item.img_path)).convert("RGB")
        if item.anomaly == 0 or not item.mask_path:
            mask = Image.fromarray(np.zeros((img.size[1], img.size[0]), dtype=np.uint8), mode="L")
        else:
            mask_full = os.path.join(self.root, item.mask_path)
            if os.path.isdir(mask_full) or not os.path.exists(mask_full):
                mask = Image.fromarray(np.zeros((img.size[1], img.size[0]), dtype=np.uint8), mode="L")
            else:
                m = np.array(Image.open(mask_full).convert("L"), dtype=np.uint8)
                m = (m > 0).astype(np.uint8) * 255
                mask = Image.fromarray(m, mode="L")
        return img, mask, item

    def sample_index(self, cls_name: str, anomaly: Optional[int] = None) -> int:
        if anomaly is None:
            candidates = self._indices_by_class.get(cls_name, [])
        else:
            candidates = self._indices_by_class_and_label.get((cls_name, int(anomaly)), [])
        if not candidates:
            raise RuntimeError(f"No candidates for cls={cls_name} anomaly={anomaly} in split={self.split}")
        return int(np.random.choice(candidates))

    def __getitem__(self, index: int) -> Dict[str, Any]:
        img, mask, item = self.load_raw(index)
        if self.caa is not None:
            img, mask = self.caa(img, mask, item)

        image_tensor = self.image_transform(img)
        mask_tensor = self.mask_transform(mask)
        mask_tensor = (mask_tensor > 0.5).float()

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "label": int(item.anomaly),
            "cls_name": item.cls_name,
            "cls_id": self.class_to_id[item.cls_name],
            "img_path": os.path.join(self.root, item.img_path),
        }
