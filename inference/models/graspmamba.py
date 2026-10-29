"""Existing GraspMamba with optional training-time target segmentation."""

import math

from torch import nn
from torch.nn import functional as F

from inference.models.segmentation import segmentation_loss


class Grasp_Mamba(nn.Module):
    def __init__(self, input_size=224, segmentation=False, lambda_seg=0.5,
                 lambda_dice=1.0, grasp_loss='mse',
                 vision_model='nvidia/MambaVision-L-1K',
                 text_model='google/siglip-base-patch16-224'):
        super().__init__()
        if any(not math.isfinite(v) or v < 0 for v in (lambda_seg, lambda_dice)):
            raise ValueError('Segmentation weights must be finite and nonnegative')
        if grasp_loss not in ('mse', 'smooth_l1'):
            raise ValueError('grasp_loss must be mse or smooth_l1')
        from inference.models.mamba.mamba_L_Siglip import GraspMamba
        self.grasp_mamba = GraspMamba(resolution=input_size, segmentation=segmentation,
                                      vision_model=vision_model, text_model=text_model)
        self.segmentation = segmentation
        self.lambda_seg = lambda_seg
        self.lambda_dice = lambda_dice
        self.grasp_loss = grasp_loss

    def forward(self, img, query, return_seg=False):
        return self.grasp_mamba(img, query, return_seg=return_seg)

    def compute_loss(self, img, yc=None, query=None, target_mask=None):
        # Both the legacy positional API and the manifest dictionary API work.
        if isinstance(img, dict):
            batch = img
            img, query = batch['image'], batch['instruction']
            yc = batch['grasp_targets']
            target_mask = batch.get('target_mask')
        if isinstance(yc, dict):
            yc = [yc[name] for name in ('pos', 'cos', 'sin', 'width')]
        if yc is None or len(yc) != 4:
            raise ValueError('Exactly four grasp targets are required')
        if self.segmentation and target_mask is None:
            raise ValueError('Segmentation training requires target_mask for every sample')
        outputs = self(img, query, return_seg=self.segmentation)
        criterion = F.mse_loss if self.grasp_loss == 'mse' else F.smooth_l1_loss
        names = ('p_loss', 'cos_loss', 'sin_loss', 'width_loss')
        losses = {name: criterion(pred, target)
                  for name, pred, target in zip(names, outputs[:4], yc)}
        grasp = sum(losses.values())
        total = grasp
        pred = dict(zip(('pos', 'cos', 'sin', 'width'), outputs[:4]))
        if self.segmentation:
            seg, bce, dice = segmentation_loss(outputs[4], target_mask, self.lambda_dice)
            losses.update(grasp_loss=grasp, seg_bce=bce, seg_dice=dice, seg_loss=seg)
            total = grasp + self.lambda_seg * seg
            pred['mask_logits'] = outputs[4]
        return {'loss': total, 'losses': losses, 'pred': pred}
