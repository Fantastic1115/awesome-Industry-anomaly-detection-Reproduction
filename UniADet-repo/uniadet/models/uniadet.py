from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class UniADetHead(nn.Module):
    """
    Learnable, fully-decoupled weights:
      - per-layer classification weights  W_cls^l in R^{2 x d}
      - per-layer segmentation weights    W_seg^l in R^{2 x d}
    """

    def __init__(self, layers: Iterable[int], embed_dim: int, tau: float = 0.07) -> None:
        super().__init__()
        self.layers: List[int] = [int(x) for x in layers]
        self.embed_dim = int(embed_dim)
        self.tau = float(tau)

        self.w_cls = nn.ParameterDict()
        self.w_seg = nn.ParameterDict()
        for layer in self.layers:
            key = str(layer)
            self.w_cls[key] = nn.Parameter(self._init_weight())
            self.w_seg[key] = nn.Parameter(self._init_weight())

    def _init_weight(self) -> torch.Tensor:
        return torch.randn(2, self.embed_dim) * 0.02

    def forward(
        self,
        layer_tokens: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
        output_size: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Args:
          layer_tokens: {layer: (cls_tokens[B,D], patch_tokens[B,N,D])}
          output_size: (H,W) to upsample segmentation probabilities.

        Returns:
          {
            "cls_logits": {layer(str): [B,2]},
            "seg_probs":  {layer(str): [B,2,H,W]}  (if output_size is not None)
          }
        """
        if output_size is None:
            raise ValueError("output_size is required to produce seg_probs")

        cls_logits: Dict[str, torch.Tensor] = {}
        seg_probs: Dict[str, torch.Tensor] = {}

        for layer in self.layers:
            key = str(layer)
            cls_tok, patch_tok = layer_tokens[layer]
            cls_tok = F.normalize(cls_tok, dim=-1)
            patch_tok = F.normalize(patch_tok, dim=-1)

            w_cls = F.normalize(self.w_cls[key], dim=-1)
            w_seg = F.normalize(self.w_seg[key], dim=-1)

            logits_cls = (cls_tok @ w_cls.t()) / self.tau
            cls_logits[key] = logits_cls

            logits_seg = (patch_tok @ w_seg.t()) / self.tau  # [B,N,2]
            probs_seg = logits_seg.softmax(dim=-1)  # [B,N,2]

            num_patches = probs_seg.shape[1]
            side = int(math.isqrt(num_patches))
            if side * side != num_patches:
                raise ValueError(f"Cannot reshape patches: N={num_patches} is not a square")

            probs_seg = probs_seg.view(probs_seg.shape[0], side, side, 2).permute(0, 3, 1, 2).contiguous()
            probs_seg = torch.nn.functional.interpolate(
                probs_seg, size=output_size, mode="bilinear", align_corners=False
            )
            seg_probs[key] = probs_seg

        return {"cls_logits": cls_logits, "seg_probs": seg_probs}

