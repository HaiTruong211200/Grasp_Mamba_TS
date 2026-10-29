"""Auxiliary target segmentation; the grasp branch never consumes masks."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class SegHead(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, 1, 1),
        )

    def forward(self, features, output_size):
        return F.interpolate(self.layers(features), size=output_size,
                             mode='bilinear', align_corners=False)


def segmentation_loss(logits, target, lambda_dice=1.0, eps=1e-6):
    """BCE on logits plus Dice reduced per sample, then averaged over batch.

    Inputs must be aligned, unpadded binary masks [B, 1, H, W]. Missing
    masks must be rejected by the loader, never replaced with background.
    """
    if not math.isfinite(lambda_dice) or lambda_dice < 0:
        raise ValueError('lambda_dice must be finite and nonnegative')
    if logits.ndim != 4 or logits.shape[1] != 1 or target.shape != logits.shape:
        raise ValueError('mask logits and targets must have identical [B,1,H,W] shapes')
    target = target.to(device=logits.device, dtype=logits.dtype)
    if not torch.isfinite(target).all() or not ((target == 0) | (target == 1)).all():
        raise ValueError('target_mask must contain only finite binary values')
    # Accumulate in float32 for stability under mixed precision.
    scores = logits.float()
    target = target.float()
    bce = F.binary_cross_entropy_with_logits(scores, target)
    p = scores.sigmoid().flatten(1)
    y = target.flatten(1)
    dice = (1 - (2 * (p * y).sum(1) + eps) /
            (p.sum(1) + y.sum(1) + eps)).mean()
    return bce + lambda_dice * dice, bce, dice
