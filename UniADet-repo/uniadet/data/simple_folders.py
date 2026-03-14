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


class ImageMaskFolderDataset(Dataset):
    """
    Generic segmentation dataset:
      root/<split>/images/*, root/<split>/masks/*
    or
      root/images/*, root/masks/*
    """

    def __init__(
        self,
        root: str,
        split: str,
        image_transform: Callable[[Image.Image], torch.Tensor],
        mask_transform: Callable[[Image.Image], torch.Tensor],
        cls_name: str,
        images_dir: Optional[str] = None,
        masks_dir: Optional[str] = None,
        caa: Optional[ClassAwareAugmentor] = None,
    ) -> None:
        self.root = os.fspath(root)
        self.split = split
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.caa = caa

        root_path = Path(self.root)
        if images_dir is None or masks_dir is None:
            images_path, masks_path = self._auto_dirs(root_path)
        else:
            images_path = root_path / images_dir
            masks_path = root_path / masks_dir

        images = _list_images(images_path)
        if not images:
            raise FileNotFoundError(f"No images found under {images_path}")

        self.class_names = [cls_name]
        self.class_to_id = {cls_name: 0}

        self.items: List[ADItem] = []
        for img_path in images:
            rel_img = img_path.relative_to(root_path).as_posix()
            mask_path = self._find_mask_for_image(masks_path, img_path)
            mask_rel = mask_path.relative_to(root_path).as_posix() if mask_path is not None else ""
            # image-level label is optional; for medical segmentation we typically report pixel metrics only.
            anomaly = 1 if mask_rel else 0
            self.items.append(ADItem(img_path=rel_img, mask_path=mask_rel, cls_name=cls_name, specie_name="", anomaly=anomaly))

        self._indices_by_class = {cls_name: list(range(len(self.items)))}
        self._indices_by_class_and_label: Dict[Tuple[str, int], List[int]] = {(cls_name, 0): [], (cls_name, 1): []}
        for idx, it in enumerate(self.items):
            self._indices_by_class_and_label[(cls_name, int(it.anomaly))].append(idx)

        if self.caa is not None:
            self.caa.bind(self)

    def _auto_dirs(self, root_path: Path) -> Tuple[Path, Path]:
        candidates = [
            (root_path / self.split / "images", root_path / self.split / "masks"),
            (root_path / self.split / "imgs", root_path / self.split / "masks"),
            (root_path / "images", root_path / "masks"),
            (root_path / "imgs", root_path / "masks"),
            (root_path / "images", root_path / "annotations"),
        ]
        for img_dir, mask_dir in candidates:
            if img_dir.is_dir() and mask_dir.is_dir():
                return img_dir, mask_dir
        raise FileNotFoundError(
            f"Cannot infer images/masks folders under {root_path}. "
            "Expected like <root>/<split>/images and <root>/<split>/masks (or images/masks at root)."
        )

    def _find_mask_for_image(self, masks_path: Path, img_path: Path) -> Optional[Path]:
        for ext in [img_path.suffix, ".png", ".jpg", ".jpeg"]:
            cand = masks_path / f"{img_path.stem}{ext}"
            if cand.exists():
                return cand
            cand2 = masks_path / f"{img_path.stem}_mask{ext}"
            if cand2.exists():
                return cand2
        return None

    def __len__(self) -> int:
        return len(self.items)

    def load_raw(self, index: int) -> Tuple[Image.Image, Image.Image, ADItem]:
        item = self.items[index]
        img = Image.open(os.path.join(self.root, item.img_path)).convert("RGB")
        if not item.mask_path:
            mask = Image.fromarray(np.zeros((img.size[1], img.size[0]), dtype=np.uint8), mode="L")
        else:
            mask_full = os.path.join(self.root, item.mask_path)
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
            raise RuntimeError(f"No candidates for cls={cls_name} anomaly={anomaly}")
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


class BinaryClassFolderDataset(Dataset):
    """
    Generic binary classification dataset based on folder names.
    Supports:
      root/<split>/<normal_dir>/*
      root/<split>/<anomaly_dir>/*
    or
      root/<normal_dir>/*
      root/<anomaly_dir>/*
    """

    def __init__(
        self,
        root: str,
        split: str,
        image_transform: Callable[[Image.Image], torch.Tensor],
        mask_transform: Callable[[Image.Image], torch.Tensor],
        cls_name: str,
        normal_dir: Optional[str] = None,
        anomaly_dir: Optional[str] = None,
        caa: Optional[ClassAwareAugmentor] = None,
    ) -> None:
        self.root = os.fspath(root)
        self.split = split
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.caa = caa

        root_path = Path(self.root)
        normal_path, anomaly_path = self._auto_dirs(root_path, normal_dir, anomaly_dir)

        normal_imgs = _list_images(normal_path)
        anomaly_imgs = _list_images(anomaly_path)
        if not normal_imgs and not anomaly_imgs:
            raise FileNotFoundError(f"No images found in {normal_path} or {anomaly_path}")

        self.class_names = [cls_name]
        self.class_to_id = {cls_name: 0}

        self.items: List[ADItem] = []
        for img_path in normal_imgs:
            rel_img = img_path.relative_to(root_path).as_posix()
            self.items.append(ADItem(img_path=rel_img, mask_path="", cls_name=cls_name, specie_name="normal", anomaly=0))
        for img_path in anomaly_imgs:
            rel_img = img_path.relative_to(root_path).as_posix()
            self.items.append(ADItem(img_path=rel_img, mask_path="", cls_name=cls_name, specie_name="anomaly", anomaly=1))

        self._indices_by_class = {cls_name: list(range(len(self.items)))}
        self._indices_by_class_and_label: Dict[Tuple[str, int], List[int]] = {(cls_name, 0): [], (cls_name, 1): []}
        for idx, it in enumerate(self.items):
            self._indices_by_class_and_label[(cls_name, int(it.anomaly))].append(idx)

        if self.caa is not None:
            self.caa.bind(self)

    def _auto_dirs(self, root_path: Path, normal_dir: Optional[str], anomaly_dir: Optional[str]) -> Tuple[Path, Path]:
        if normal_dir and anomaly_dir:
            cand_a = root_path / self.split / normal_dir
            cand_b = root_path / self.split / anomaly_dir
            if cand_a.is_dir() and cand_b.is_dir():
                return cand_a, cand_b
            cand_a = root_path / normal_dir
            cand_b = root_path / anomaly_dir
            if cand_a.is_dir() and cand_b.is_dir():
                return cand_a, cand_b
            raise FileNotFoundError(f"Cannot find provided dirs: {normal_dir}, {anomaly_dir} under {root_path}")

        # Heuristic: choose two subdirs by name keywords.
        keywords_normal = {"normal", "good", "ok", "negative", "no", "benign", "healthy"}
        keywords_anomaly = {"anomaly", "abnormal", "bad", "ng", "positive", "yes", "tumor", "lesion", "malignant"}

        candidates = []
        split_root = root_path / self.split
        if split_root.is_dir():
            candidates = [p for p in split_root.iterdir() if p.is_dir()]
        else:
            candidates = [p for p in root_path.iterdir() if p.is_dir()]

        norm = None
        anom = None
        for p in candidates:
            name = p.name.lower()
            if any(k in name for k in keywords_normal) and norm is None:
                norm = p
            if any(k in name for k in keywords_anomaly) and anom is None:
                anom = p

        if norm is None or anom is None:
            # fallback: common (0/1)
            for p in candidates:
                if p.name == "0" and norm is None:
                    norm = p
                if p.name == "1" and anom is None:
                    anom = p

        if norm is None or anom is None:
            raise FileNotFoundError(
                f"Cannot infer normal/anomaly subfolders under {split_root if split_root.is_dir() else root_path}. "
                "Provide --dataset meta with meta.json or add explicit folder names support."
            )
        return norm, anom

    def __len__(self) -> int:
        return len(self.items)

    def load_raw(self, index: int) -> Tuple[Image.Image, Image.Image, ADItem]:
        item = self.items[index]
        img = Image.open(os.path.join(self.root, item.img_path)).convert("RGB")
        mask = Image.fromarray(np.zeros((img.size[1], img.size[0]), dtype=np.uint8), mode="L")
        return img, mask, item

    def sample_index(self, cls_name: str, anomaly: Optional[int] = None) -> int:
        if anomaly is None:
            candidates = self._indices_by_class.get(cls_name, [])
        else:
            candidates = self._indices_by_class_and_label.get((cls_name, int(anomaly)), [])
        if not candidates:
            raise RuntimeError(f"No candidates for cls={cls_name} anomaly={anomaly}")
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

