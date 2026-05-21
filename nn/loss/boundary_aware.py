"""
Custom loss function to penalize boundary offsets and over-segmentation.

Core ideas:
- Add extra penalty to False Positive regions
  (predicted as foreground but with no corresponding GT)
- Add a soft penalty for Offset (inaccurate boundaries)
- Combine with standard CrossEntropy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BoundaryAwareLoss(nn.Module):
    """
    CrossEntropy + boundary-aware penalty.

    Parameters
    ----------
    alpha : float
        Weight of the CrossEntropy term
    beta : float
        Weight of the boundary penalty term (penalizes Offset)
    gamma : float
        Weight of the False Positive penalty
        (penalizes foreground predictions on GT background)
    # delta : float
    #     Weight of the over-segmentation penalty
    #     (penalizes frequent frame-to-frame prediction switching)
    #     - found ineffective in experiments and disabled
    background_index : int
        Class index of the background class, usually 0
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.2,
        gamma: float = 0.5,
        # delta: float = 0.3,  # ineffective in experiments, disabled
        background_index: int = 0,
        weight: torch.Tensor | None = None,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        # self.delta = delta  # ineffective in experiments, disabled
        self.background_index = background_index
        self.ce = nn.CrossEntropyLoss(weight=weight)

    def forward(
        self,
        output: torch.Tensor,   # (batch, n_classes, time)
        target: torch.Tensor,   # (batch, time)
    ) -> torch.Tensor:

        # 1. Standard CrossEntropy
        ce_loss = self.ce(output, target)

        # 2. Boundary-aware penalty
        boundary_mask = self._get_boundary_mask(target)

        if boundary_mask.sum() > 0:
            boundary_loss = F.cross_entropy(
                output.permute(0, 2, 1)[boundary_mask],
                target[boundary_mask],
                reduction='mean'
            )
        else:
            boundary_loss = torch.tensor(0.0, device=output.device)

        # 3. False Positive penalty
        pred_classes = output.argmax(dim=1)

        gt_is_bg = (target == self.background_index)
        pred_is_fg = (pred_classes != self.background_index)

        fp_mask = gt_is_bg & pred_is_fg

        if fp_mask.sum() > 0:
            fp_loss = F.cross_entropy(
                output.permute(0, 2, 1)[fp_mask],
                target[fp_mask],
                reduction='mean'
            )
        else:
            fp_loss = torch.tensor(0.0, device=output.device)

        # 4. transition_loss:
        # experimentally ineffective and disabled
        # probs = F.softmax(output, dim=1)
        # prob_diff = (probs[:, :, 1:] - probs[:, :, :-1]).abs().sum(dim=1)
        # transition_loss = prob_diff.mean()

        total = (
            self.alpha * ce_loss
            + self.beta * boundary_loss
            + self.gamma * fp_loss
            # + self.delta * transition_loss
        )

        return total

    @staticmethod
    def _get_boundary_mask(
        target: torch.Tensor,
        radius: int = 2
    ) -> torch.Tensor:
        """
        Return a boolean mask covering ±radius frames
        around GT label boundaries.

        Boundary = positions where adjacent frame labels differ.
        """

        diff = (target[:, 1:] != target[:, :-1])  # (batch, time-1)

        mask = torch.zeros_like(target, dtype=torch.bool)

        for r in range(radius + 1):

            if r == 0:
                mask[:, :-1] |= diff
                mask[:, 1:]  |= diff

            else:
                mask[:, :-(r + 1)] |= diff[:, r:]
                mask[:, (r + 1):]  |= diff[:, :-r]

        return mask