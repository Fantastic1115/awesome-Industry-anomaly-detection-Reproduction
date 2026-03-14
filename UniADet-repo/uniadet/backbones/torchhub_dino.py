from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn


class TorchHubDINOBackbone(nn.Module):
    """
    DINOv2/DINOv3 backbone wrapper using torch.hub.

    Notes:
    - The paper uses 1-based block numbers (e.g., 24-th block). TorchHub DINO models
      use 0-based indices in get_intermediate_layers. We convert internally.
    """

    def __init__(self, repo: str, model_name: str, device: torch.device) -> None:
        super().__init__()
        self.repo = repo
        self.model_name = model_name
        self.model = torch.hub.load(repo, model_name)
        self.model.eval()
        self.model.requires_grad_(False)
        self.model.to(device)

        self.embed_dim = int(getattr(self.model, "embed_dim", getattr(self.model, "dim", 0)))
        if not self.embed_dim:
            self.embed_dim = int(self.model.norm.weight.shape[0])
        self.patch_size = int(getattr(self.model, "patch_size", 16))

    @torch.no_grad()
    def extract_tokens(self, images: torch.Tensor, layers: List[int]) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Returns:
          {layer_index(1-based): (cls_tokens[B,D], patch_tokens[B,N,D])}
        """
        blocks_0 = [int(l - 1) for l in layers]
        outputs = self.model.get_intermediate_layers(images, n=blocks_0, return_class_token=True, norm=True)
        tokens_by_layer: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        for layer, out in zip(layers, outputs):
            patch_tokens, cls_tokens = out
            tokens_by_layer[int(layer)] = (cls_tokens, patch_tokens)
        return tokens_by_layer

