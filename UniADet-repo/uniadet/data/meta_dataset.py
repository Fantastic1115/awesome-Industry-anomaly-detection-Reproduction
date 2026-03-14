from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .caa import ClassAwareAugmentor
from .types import ADItem


class MetaJsonDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        image_transform: Callable[[Image.Image], torch.Tensor],
        mask_transform: Callable[[Image.Image], torch.Tensor],
        caa: Optional[ClassAwareAugmentor] = None,
    ) -> None:
        self.root = root
        self.split = split
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.caa = caa

        meta_path = os.path.join(root, "meta.json")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        if split not in meta:
            raise KeyError(f"Split '{split}' not found in meta.json (available: {list(meta.keys())})")

        per_class = meta[split]
        self.class_names = sorted(list(per_class.keys()))
        self.class_to_id = {name: i for i, name in enumerate(self.class_names)}

        self.items: List[ADItem] = []
        for cls_name in self.class_names:
            for it in per_class[cls_name]:
                self.items.append(
                    ADItem(
                        img_path=it["img_path"],
                        mask_path=it.get("mask_path", ""),
                        cls_name=it["cls_name"],
                        specie_name=it.get("specie_name", ""),
                        anomaly=int(it["anomaly"]),
                    )
                )

        self._indices_by_class: Dict[str, List[int]] = {c: [] for c in self.class_names}
        self._indices_by_class_and_label: Dict[Tuple[str, int], List[int]] = {}
        for idx, item in enumerate(self.items):
            self._indices_by_class[item.cls_name].append(idx)
            self._indices_by_class_and_label.setdefault((item.cls_name, item.anomaly), []).append(idx)

        if self.caa is not None:
            self.caa.bind(self)

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
