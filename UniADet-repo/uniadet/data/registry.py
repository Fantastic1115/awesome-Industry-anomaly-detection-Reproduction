from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple

import torch

from .caa import ClassAwareAugmentor
from .meta_dataset import MetaJsonDataset
from .mvtec_style import MVTecStyleDataset
from .realiad import RealIADDataset, discover_realiad_paths
from .simple_folders import BinaryClassFolderDataset, ImageMaskFolderDataset


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    kind: str  # "auto" | "meta" | "mvtec_style" | "realiad"
    good_name: str = "good"
    gt_dir_name: str = "ground_truth"
    default_metrics: str = "image-pixel"


_SPECS = {
    # Industrial
    "mvtec": DatasetSpec(name="mvtec", kind="mvtec_style", good_name="good", gt_dir_name="ground_truth", default_metrics="image-pixel"),
    "visa": DatasetSpec(name="visa", kind="mvtec_style", good_name="good", gt_dir_name="ground_truth", default_metrics="image-pixel"),
    "btad": DatasetSpec(name="btad", kind="mvtec_style", good_name="ok", gt_dir_name="ground_truth", default_metrics="image-pixel"),
    "dtd": DatasetSpec(name="dtd", kind="mvtec_style", good_name="good", gt_dir_name="ground_truth", default_metrics="image-pixel"),
    "ksdd": DatasetSpec(name="ksdd", kind="mvtec_style", good_name="good", gt_dir_name="ground_truth", default_metrics="image-pixel"),
    "real-iad": DatasetSpec(name="real-iad", kind="realiad", default_metrics="image-pixel"),
    "real_iad": DatasetSpec(name="real-iad", kind="realiad", default_metrics="image-pixel"),
    "realiad": DatasetSpec(name="real-iad", kind="realiad", default_metrics="image-pixel"),
    # Medical (often provided via meta.json in practice)
    "headct": DatasetSpec(name="headct", kind="auto", default_metrics="image"),
    "brainmri": DatasetSpec(name="brainmri", kind="auto", default_metrics="image"),
    "br35h": DatasetSpec(name="br35h", kind="auto", default_metrics="image"),
    "isic": DatasetSpec(name="isic", kind="auto", default_metrics="pixel"),
    "colondb": DatasetSpec(name="colondb", kind="auto", default_metrics="pixel"),
    "clinicdb": DatasetSpec(name="clinicdb", kind="auto", default_metrics="pixel"),
    "kvasir": DatasetSpec(name="kvasir", kind="auto", default_metrics="pixel"),
    "endo": DatasetSpec(name="endo", kind="auto", default_metrics="pixel"),
}


def normalize_dataset_name(name: str) -> str:
    name = (name or "auto").strip().lower()
    return name


def resolve_spec(name: str) -> DatasetSpec:
    name = normalize_dataset_name(name)
    if name == "auto":
        return DatasetSpec(name="auto", kind="auto")
    if name == "meta":
        return DatasetSpec(name="meta", kind="meta")
    if name in {"folder", "mvtec-style", "mvtec_style"}:
        return DatasetSpec(name=name, kind="auto")
    if name in _SPECS:
        return _SPECS[name]
    raise ValueError(f"Unknown dataset '{name}'. Supported: {sorted(set(_SPECS.keys()) | {'auto','meta','folder'})}")


def _detect_mvtec_style_root(root: str) -> bool:
    root_path = Path(os.fspath(root))
    if (root_path / "train").is_dir() and (root_path / "test").is_dir():
        return True
    for p in root_path.iterdir():
        if p.is_dir() and (p / "train").is_dir() and (p / "test").is_dir():
            return True
    return False


def _detect_realiad_root(root: str) -> bool:
    try:
        discover_realiad_paths(root)
        return True
    except Exception:
        return False


def _auto_good_name(root: str, candidates: Sequence[str] = ("good", "ok", "normal")) -> str:
    root_path = Path(os.fspath(root))
    # Try single class first
    train_dir = root_path / "train"
    if train_dir.is_dir():
        for c in candidates:
            if (train_dir / c).is_dir():
                return c
    # Try multi-class
    for cat in root_path.iterdir():
        if not cat.is_dir():
            continue
        train_dir = cat / "train"
        if not train_dir.is_dir():
            continue
        for c in candidates:
            if (train_dir / c).is_dir():
                return c
    return "good"


def build_dataset(
    *,
    dataset: str,
    root: str,
    split: str,
    image_transform: Callable,
    mask_transform: Callable,
    caa: Optional[ClassAwareAugmentor] = None,
    class_names: Optional[Sequence[str]] = None,
):
    """
    Build a dataset with automatic adaptation. Priority:
      1) If meta.json exists or dataset=='meta' -> MetaJsonDataset
      2) If dataset spec says realiad -> RealIADDataset
      3) If dataset spec says mvtec_style -> MVTecStyleDataset
      4) If dataset=='auto' and folder looks like Real-IAD -> RealIADDataset
      5) If dataset=='auto' and folder looks mvtec-style -> MVTecStyleDataset
    """
    spec = resolve_spec(dataset)
    meta_path = Path(os.fspath(root)) / "meta.json"
    force_folder = normalize_dataset_name(dataset) in {"folder", "mvtec-style", "mvtec_style"}
    if meta_path.is_file() and not force_folder:
        return MetaJsonDataset(root=root, split=split, image_transform=image_transform, mask_transform=mask_transform, caa=caa)
    if spec.kind == "meta" and meta_path.is_file():
        return MetaJsonDataset(root=root, split=split, image_transform=image_transform, mask_transform=mask_transform, caa=caa)

    if spec.kind == "realiad":
        return RealIADDataset(
            root=root,
            split=split,
            image_transform=image_transform,
            mask_transform=mask_transform,
            caa=caa,
            categories=class_names,
        )

    if spec.kind == "mvtec_style":
        return MVTecStyleDataset(
            root=root,
            split=split,
            image_transform=image_transform,
            mask_transform=mask_transform,
            good_name=spec.good_name,
            gt_dir_name=spec.gt_dir_name,
            caa=caa,
            class_names=class_names,
        )

    if spec.kind == "auto":
        if _detect_realiad_root(root):
            return RealIADDataset(
                root=root,
                split=split,
                image_transform=image_transform,
                mask_transform=mask_transform,
                caa=caa,
                categories=class_names,
            )

        if _detect_mvtec_style_root(root):
            good_name = _auto_good_name(root)
            return MVTecStyleDataset(
                root=root,
                split=split,
                image_transform=image_transform,
                mask_transform=mask_transform,
                good_name=good_name,
                gt_dir_name="ground_truth",
                caa=caa,
                class_names=class_names,
            )

        # Heuristics for medical datasets without meta.json:
        # - segmentation: images + masks folders
        # - classification: binary folders
        name = normalize_dataset_name(dataset)
        if name in {"isic", "colondb", "clinicdb", "kvasir", "endo"}:
            return ImageMaskFolderDataset(
                root=root,
                split=split,
                image_transform=image_transform,
                mask_transform=mask_transform,
                cls_name=name,
                caa=caa,
            )
        if name in {"headct", "brainmri", "br35h"}:
            return BinaryClassFolderDataset(
                root=root,
                split=split,
                image_transform=image_transform,
                mask_transform=mask_transform,
                cls_name=name,
                caa=caa,
            )

    raise FileNotFoundError(
        f"Cannot auto-adapt dataset at '{root}'. "
        "Provide a meta.json (meta format) or use a supported MVTec-style layout."
    )


def default_metrics_for_dataset(dataset: str) -> str:
    spec = resolve_spec(dataset)
    if spec.name == "auto":
        return "image-pixel"
    return spec.default_metrics
