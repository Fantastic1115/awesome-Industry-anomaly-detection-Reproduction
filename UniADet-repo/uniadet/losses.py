from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    """
    Multi-class focal loss that expects probability inputs (after softmax),
    following the common implementation used in AD baselines (e.g., AnomalyCLIP).
    """

    def __init__(
        self,
        apply_nonlin: Optional[callable] = None,
        alpha=None,
        gamma: float = 2.0,
        balance_index: int = 0,
        smooth: float = 1e-5,
        size_average: bool = True,
    ) -> None:
        super().__init__()
        self.apply_nonlin = apply_nonlin
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.smooth = smooth
        self.size_average = size_average

        if self.smooth < 0 or self.smooth > 1.0:
            raise ValueError("smooth should be in [0, 1]")

    def forward(self, logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.apply_nonlin is not None:
            logit = self.apply_nonlin(logit)

        num_class = logit.shape[1]
        if logit.dim() > 2:
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))

        target = torch.squeeze(target, 1)
        target = target.view(-1, 1)

        alpha = self.alpha
        if alpha is None:
            alpha = torch.ones(num_class, 1)
        elif isinstance(alpha, (list, np.ndarray)):
            if len(alpha) != num_class:
                raise ValueError("alpha length must match num_class")
            alpha = torch.FloatTensor(alpha).view(num_class, 1)
            alpha = alpha / alpha.sum()
        elif isinstance(alpha, float):
            alpha_tensor = torch.ones(num_class, 1)
            alpha_tensor = alpha_tensor * (1 - alpha)
            alpha_tensor[self.balance_index] = alpha
            alpha = alpha_tensor
        else:
            raise TypeError("Unsupported alpha type")

        if alpha.device != logit.device:
            alpha = alpha.to(logit.device)

        idx = target.cpu().long()
        one_hot_key = torch.zeros(target.size(0), num_class, device=logit.device)
        one_hot_key = one_hot_key.scatter_(1, idx.to(logit.device), 1)

        if self.smooth:
            one_hot_key = torch.clamp(one_hot_key, self.smooth / (num_class - 1), 1.0 - self.smooth)

        pt = (one_hot_key * logit).sum(1) + self.smooth
        logpt = pt.log()

        alpha = alpha[idx].squeeze().to(logit.device)
        loss = -1.0 * alpha * torch.pow((1 - pt), self.gamma) * logpt
        return loss.mean() if self.size_average else loss.sum()


class BinaryDiceLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n = targets.size(0)
        smooth = 1.0
        input_flat = input.view(n, -1)
        targets_flat = targets.view(n, -1)
        intersection = input_flat * targets_flat
        dice = (2 * intersection.sum(1) + smooth) / (input_flat.sum(1) + targets_flat.sum(1) + smooth)
        return 1.0 - dice.mean()

