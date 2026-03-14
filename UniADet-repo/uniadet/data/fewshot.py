from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class FewShotMemoryBank:
    """
    Multi-layer normal memory bank for few-shot inference (paper Eq.9-12).
    Stored per class and per layer as normalized patch tokens.
    """

    layers: List[int]
    bank: Dict[str, Dict[int, torch.Tensor]]  # cls_name -> layer -> [N_mem, D]

    @classmethod
    def build(
        cls,
        backbone,
        dataset,
        layers: List[int],
        shots: int,
        device: torch.device,
        seed: int = 0,
    ) -> "FewShotMemoryBank":
        bank: Dict[str, Dict[int, torch.Tensor]] = {}

        indices_by_class: Dict[str, List[int]] = {c: [] for c in dataset.class_names}
        for idx in range(len(dataset)):
            item = dataset.items[idx]
            if int(item.anomaly) == 0:
                indices_by_class[item.cls_name].append(idx)

        rng = np.random.RandomState(seed)

        for cls_name in dataset.class_names:
            candidates = indices_by_class.get(cls_name, [])
            if not candidates:
                continue
            rng.shuffle(candidates)
            selected = candidates[: int(shots)]
            if not selected:
                continue

            per_layer_feats: Dict[int, List[torch.Tensor]] = {l: [] for l in layers}
            for idx in selected:
                sample = dataset[idx]
                image = sample["image"].unsqueeze(0).to(device)
                with torch.no_grad():
                    tokens = backbone.extract_tokens(image, layers)
                for l in layers:
                    _, patch = tokens[l]  # [1,N,D]
                    patch = F.normalize(patch, dim=-1)  # [1,N,D]
                    per_layer_feats[l].append(patch.squeeze(0).detach().cpu())

            bank[cls_name] = {}
            for l in layers:
                feats = torch.cat(per_layer_feats[l], dim=0)  # [K*N, D]
                feats = F.normalize(feats, dim=-1)
                bank[cls_name][l] = feats.to(device)

        return cls(layers=list(layers), bank=bank)

    def score_map_for_class(
        self,
        cls_name: str,
        layer_tokens: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
        output_size: Tuple[int, int],
    ) -> np.ndarray:
        """
        Compute few-shot anomaly map for a single image (B=1).
        Returns HxW numpy array.
        """
        if cls_name not in self.bank:
            return np.zeros(output_size, dtype=np.float32)

        maps = []
        for l in self.layers:
            _, patch = layer_tokens[l]  # [1,N,D]
            patch = F.normalize(patch, dim=-1).squeeze(0)  # [N,D]
            mem = self.bank[cls_name][l]  # [M,D]

            sim = patch @ mem.t()  # [N,M]
            max_sim, _ = sim.max(dim=1)  # [N]
            dist = 1.0 - max_sim
            maps.append(dist.detach().cpu())

        dist_mean = torch.stack(maps, dim=0).mean(dim=0)  # [N]
        num_patches = int(dist_mean.numel())
        side = int(np.sqrt(num_patches))
        if side * side != num_patches:
            raise ValueError(f"Cannot reshape patches: N={num_patches} is not a square")

        dist_grid = dist_mean.view(1, 1, side, side)
        dist_grid = torch.nn.functional.interpolate(dist_grid, size=output_size, mode="bilinear", align_corners=False)
        return dist_grid[0, 0].numpy().astype(np.float32)
