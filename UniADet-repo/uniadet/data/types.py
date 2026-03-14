from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ADItem:
    img_path: str
    mask_path: str
    cls_name: str
    specie_name: str
    anomaly: int

