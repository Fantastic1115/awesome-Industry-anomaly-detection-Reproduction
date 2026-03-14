from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class HFCLIPBackbone(nn.Module):
    """
    CLIP vision encoder wrapper (HuggingFace) that exposes CLS token and patch tokens
    for specified transformer blocks (1-based indices as in the paper).
    """

    def __init__(self, model_name: str, device: torch.device, image_size: Optional[int] = None) -> None:
        super().__init__()
        from transformers import CLIPVisionModel  # lazy import

        self.model_name = model_name
        self.model = CLIPVisionModel.from_pretrained(model_name)
        self.model.eval()
        self.model.requires_grad_(False)
        self.model.to(device)

        self.embed_dim = int(self.model.config.hidden_size)
        self.patch_size = int(self.model.config.patch_size)

        if image_size is not None:
            self._resize_position_embeddings(image_size=image_size)

    def _resize_position_embeddings(self, image_size: int) -> None:
        """
        HF CLIP uses fixed absolute position embeddings. UniADet uses 518x518 (ViT-L/14),
        which requires interpolating position embeddings beyond the original 336px grid.
        """
        if image_size % self.patch_size != 0:
            raise ValueError(f"image_size={image_size} must be divisible by patch_size={self.patch_size}")

        new_grid = image_size // self.patch_size
        new_num_patches = new_grid * new_grid
        new_num_positions = new_num_patches + 1

        embeddings = self.model.vision_model.embeddings
        pos_embed: torch.Tensor = embeddings.position_embedding.weight.data  # [num_positions, dim]
        old_num_positions, dim = pos_embed.shape
        if old_num_positions == new_num_positions:
            return

        old_num_patches = old_num_positions - 1
        old_grid = int(math.isqrt(old_num_patches))
        if old_grid * old_grid != old_num_patches:
            raise RuntimeError(f"Unexpected CLIP pos_embed length: {old_num_positions}")

        cls_pos = pos_embed[:1]  # [1, dim]
        patch_pos = pos_embed[1:]  # [old_patches, dim]
        patch_pos = patch_pos.reshape(1, old_grid, old_grid, dim).permute(0, 3, 1, 2)  # [1,dim,gh,gw]
        patch_pos = F.interpolate(patch_pos, size=(new_grid, new_grid), mode="bicubic", align_corners=False)
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(new_num_patches, dim)
        new_pos_embed = torch.cat([cls_pos, patch_pos], dim=0)  # [new_positions, dim]

        # Replace embedding layer and position ids buffer.
        new_position_embedding = nn.Embedding(new_num_positions, dim)
        new_position_embedding.weight.data.copy_(new_pos_embed)
        new_position_embedding.requires_grad_(False)
        embeddings.position_embedding = new_position_embedding.to(pos_embed.device)
        embeddings.num_positions = new_num_positions
        embeddings.position_ids = torch.arange(new_num_positions, device=pos_embed.device).expand((1, -1))

        # Keep config in sync for downstream usage/debugging.
        self.model.config.image_size = int(image_size)
        self.model.config.max_position_embeddings = int(new_num_positions)

    @torch.no_grad()
    def extract_tokens(self, images: torch.Tensor, layers: List[int]) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Returns:
          {layer_index: (cls_tokens[B,D], patch_tokens[B,N,D])}
        """
        outputs = self.model(images, output_hidden_states=True, return_dict=True)
        hidden_states = outputs.hidden_states  # tuple: [embeddings] + per-layer

        tokens_by_layer: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        for layer in layers:
            if layer < 0 or layer >= len(hidden_states):
                raise ValueError(f"Layer {layer} out of range for hidden_states len={len(hidden_states)}")
            tokens = hidden_states[layer]  # [B,1+N,D]
            cls_tokens = tokens[:, 0, :]
            patch_tokens = tokens[:, 1:, :]
            tokens_by_layer[int(layer)] = (cls_tokens, patch_tokens)
        return tokens_by_layer
