from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .caa import ClassAwareAugmentor
from .types import ADItem


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class RealIADPaths:
    root: Path
    image_root: Path
    json_root: Path


def _first_existing_dir(root: Path, candidates: Sequence[str]) -> Optional[Path]:
    for c in candidates:
        p = root / c
        if p.is_dir():
            return p
    return None


def _find_realiad_json_root(root: Path) -> Optional[Path]:
    """
    Supports both layouts:
      root/realiad_jsons/*.json
      root/realiad_jsons/realiad_jsons/*.json  (zip with nested folder)
    and similarly for *_sv, *_fuiad_* variants.
    """
    # Prefer base metadata if available.
    candidates = [
        "realiad_jsons",
        "realiad_jsons/realiad_jsons",
        "realiad_jsons_sv",
        "realiad_jsons_sv/realiad_jsons_sv",
    ]
    # Add any fuiad folders we can see (0.0/0.1/0.2/0.4 etc).
    for p in sorted(root.glob("realiad_jsons_fuiad*")):
        if p.is_dir():
            candidates.append(p.name)
            candidates.append(f"{p.name}/{p.name}")

    for c in candidates:
        d = root / c
        if d.is_dir() and any(x.suffix.lower() == ".json" for x in d.glob("*.json")):
            return d
    return None


def _find_realiad_image_root(root: Path) -> Optional[Path]:
    return _first_existing_dir(root, ["realiad_1024", "realiad_512", "realiad_256", "realiad_raw"])


def discover_realiad_paths(root: str) -> RealIADPaths:
    root_path = Path(os.fspath(root)).expanduser().resolve()
    if not root_path.is_dir():
        raise FileNotFoundError(root_path)

    image_root = _find_realiad_image_root(root_path)
    if image_root is None:
        raise FileNotFoundError(
            f"Cannot find Real-IAD image folder under {root_path}. Expected one of: realiad_1024/realiad_512/realiad_256/realiad_raw"
        )

    json_root = _find_realiad_json_root(root_path)
    if json_root is None:
        raise FileNotFoundError(
            f"Cannot find Real-IAD json folder under {root_path}. Expected: realiad_jsons (or nested realiad_jsons/realiad_jsons)"
        )

    return RealIADPaths(root=root_path, image_root=image_root, json_root=json_root)


def _norm_str(x: Any) -> str:
    if x is None:
        return ""
    return str(x).replace("\\", "/").strip()


def _is_image_path(s: str) -> bool:
    suf = Path(s).suffix.lower()
    return suf in IMAGE_EXTS


def _to_label_index(x: Any) -> Optional[int]:
    if x is None:
        return None
    if isinstance(x, bool):
        return int(bool(x))
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, float):
        return int(x)
    s = str(x).strip().lower()
    if s in {"0", "ok", "good", "normal", "negative", "neg", "no"}:
        return 0
    if s in {"1", "ng", "bad", "abnormal", "anomaly", "anomalous", "positive", "pos", "yes", "defect"}:
        return 1
    return None


def _norm_split(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip().lower()
    if s in {"train", "training"}:
        return "train"
    if s in {"test", "testing"}:
        return "test"
    if s in {"val", "valid", "validation"}:
        return "val"
    return s


def _infer_mask_from_image_rel(image_rel: str) -> str:
    p = Path(image_rel)
    if p.suffix.lower() == ".jpg" or p.suffix.lower() == ".jpeg":
        return str(p.with_suffix(".png")).replace("\\", "/")
    # try common naming
    return str(p.with_name(p.stem + "_mask.png")).replace("\\", "/")


def _flatten_realiad_json(obj: Any, *, split: Optional[str] = None, label: Optional[int] = None) -> Iterable[Dict[str, Any]]:
    """
    Yield record-like dicts from unknown Real-IAD json structures.
    Tries to keep track of split/label context from nesting keys.
    """
    if isinstance(obj, dict):
        # Record candidate: dict has explicit image path.
        image_keys = ["image_path", "img_path", "image", "img", "path", "file", "filename"]
        mask_keys = ["mask_path", "mask", "gt_mask_path", "gt", "maskfile", "label_path"]
        split_keys = ["split", "phase", "set", "mode"]
        label_keys = ["label_index", "label", "anomaly", "is_anomaly", "gt_label"]

        image_path = None
        for k in image_keys:
            v = obj.get(k)
            if isinstance(v, str) and _is_image_path(v):
                image_path = v
                break

        if image_path is not None:
            rec: Dict[str, Any] = {"image_path": image_path}
            for k in mask_keys:
                v = obj.get(k)
                if isinstance(v, str) and v:
                    rec["mask_path"] = v
                    break
            for k in split_keys:
                if k in obj:
                    rec["split"] = obj.get(k)
                    break
            for k in label_keys:
                if k in obj:
                    rec["label"] = obj.get(k)
                    break
            if split is not None and "split" not in rec:
                rec["split"] = split
            if label is not None and "label" not in rec:
                rec["label"] = label
            yield rec
            return

        for k, v in obj.items():
            next_split = split
            next_label = label
            k_norm = str(k).strip().lower()
            if k_norm in {"train", "test", "val", "valid", "validation"}:
                next_split = _norm_split(k_norm)
            if k_norm in {"good", "ok", "normal"}:
                next_label = 0
            if k_norm in {"ng", "bad", "anomaly", "anomalous", "defect", "abnormal"}:
                next_label = 1
            yield from _flatten_realiad_json(v, split=next_split, label=next_label)
        return

    if isinstance(obj, list):
        for it in obj:
            yield from _flatten_realiad_json(it, split=split, label=label)
        return

    if isinstance(obj, str):
        s = obj.strip()
        if _is_image_path(s):
            yield {"image_path": s, "split": split, "label": label}
        return


def _resolve_rel_path(
    *,
    root: Path,
    image_root: Path,
    category: str,
    path_str: str,
) -> str:
    p = _norm_str(path_str)
    if not p:
        return ""

    # If absolute, try to make it relative.
    try:
        abs_path = Path(p)
        if abs_path.is_absolute():
            try:
                return abs_path.relative_to(root).as_posix()
            except Exception:
                return abs_path.name
    except Exception:
        pass

    # Strip leading "./"
    p = re.sub(r"^\./+", "", p)

    # If it already contains a known image root folder, keep suffix from that.
    for marker in ["realiad_1024/", "realiad_512/", "realiad_256/", "realiad_raw/"]:
        idx = p.find(marker)
        if idx >= 0:
            p2 = p[idx:]
            if (root / p2).exists():
                return p2

    # Candidate 1: path under image_root directly.
    if (image_root / p).exists():
        return (image_root.name + "/" + p).replace("\\", "/")

    # Candidate 2: path under image_root/category
    if not p.startswith(category + "/") and (image_root / category / p).exists():
        return (image_root.name + "/" + category + "/" + p).replace("\\", "/")
    if p.startswith(category + "/") and (image_root / p).exists():
        return (image_root.name + "/" + p).replace("\\", "/")

    # Candidate 3: maybe already relative to root.
    if (root / p).exists():
        return p

    # Fallback: keep as-is; caller may still open via root/image_root/category.
    if not p.startswith(image_root.name + "/"):
        return (image_root.name + "/" + category + "/" + p).replace("\\", "/")
    return p


class RealIADDataset(Dataset):
    """
    Real-IAD loader that reads json split files and resolves images/masks.

    Expected root structure (official guidance):
      root/realiad_1024/<object>/*.jpg, *.png
      root/realiad_jsons/<object>.json
    Also supports nested json folder after unzip: root/realiad_jsons/realiad_jsons/<object>.json
    """

    def __init__(
        self,
        root: str,
        split: str,
        image_transform: Callable[[Image.Image], torch.Tensor],
        mask_transform: Callable[[Image.Image], torch.Tensor],
        caa: Optional[ClassAwareAugmentor] = None,
        categories: Optional[Sequence[str]] = None,
    ) -> None:
        self.paths = discover_realiad_paths(root)
        self.root = os.fspath(self.paths.root)
        self.split = split
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.caa = caa

        json_files = sorted(self.paths.json_root.glob("*.json"))
        if not json_files:
            raise FileNotFoundError(f"No json files found in {self.paths.json_root}")

        allow = None if categories is None else {str(c) for c in categories}
        if allow is not None:
            json_files = [p for p in json_files if p.stem in allow]
            if not json_files:
                raise ValueError(f"No requested categories found in {self.paths.json_root}: {sorted(allow)}")

        self.class_names = sorted([p.stem for p in json_files])
        self.class_to_id = {name: i for i, name in enumerate(self.class_names)}

        split_norm = _norm_split(split) or split
        items: List[ADItem] = []

        for jf in json_files:
            category = jf.stem
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
            for rec in _flatten_realiad_json(data):
                rec_split = _norm_split(rec.get("split")) if "split" in rec else None
                if rec_split is not None and rec_split != split_norm:
                    continue
                if rec_split is None and split_norm not in {"train", "test", "val"}:
                    # Unknown json structure; user provided custom split, keep all.
                    pass

                img_p = _norm_str(rec.get("image_path"))
                if not img_p:
                    continue

                label_index = _to_label_index(rec.get("label"))

                img_rel = _resolve_rel_path(
                    root=self.paths.root,
                    image_root=self.paths.image_root,
                    category=category,
                    path_str=img_p,
                )

                mask_rel = _norm_str(rec.get("mask_path"))
                if mask_rel:
                    mask_rel = _resolve_rel_path(
                        root=self.paths.root,
                        image_root=self.paths.image_root,
                        category=category,
                        path_str=mask_rel,
                    )
                elif label_index == 1:
                    # Infer mask path from image.
                    mask_rel = _infer_mask_from_image_rel(img_rel)
                    if not (self.paths.root / mask_rel).exists():
                        # Try without image_root prefix if needed.
                        candidate = _infer_mask_from_image_rel(img_p)
                        candidate_rel = _resolve_rel_path(
                            root=self.paths.root,
                            image_root=self.paths.image_root,
                            category=category,
                            path_str=candidate,
                        )
                        if (self.paths.root / candidate_rel).exists():
                            mask_rel = candidate_rel
                        else:
                            mask_rel = ""

                if label_index is None:
                    # Fallback: infer from mask existence.
                    label_index = 1 if (mask_rel and (self.paths.root / mask_rel).exists()) else 0

                items.append(
                    ADItem(
                        img_path=img_rel,
                        mask_path=mask_rel,
                        cls_name=category,
                        specie_name="",
                        anomaly=int(label_index),
                    )
                )

        if not items:
            raise RuntimeError(
                f"No samples found for split='{split}' under {self.paths.json_root}. "
                "If your json doesn't store split info, try a different --split or provide meta.json."
            )

        self.items = items

        self._indices_by_class: Dict[str, List[int]] = {c: [] for c in self.class_names}
        self._indices_by_class_and_label: Dict[Tuple[str, int], List[int]] = {}
        for idx, it in enumerate(self.items):
            self._indices_by_class[it.cls_name].append(idx)
            self._indices_by_class_and_label.setdefault((it.cls_name, int(it.anomaly)), []).append(idx)

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

