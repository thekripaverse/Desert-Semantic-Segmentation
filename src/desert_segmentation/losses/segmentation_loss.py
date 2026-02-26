import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class SegmentationLoss(nn.Module):
    """
    Hybrid segmentation loss combining weighted Cross Entropy and Dice Loss.
    """

    def __init__(self, class_weights: torch.Tensor):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.dice = smp.losses.DiceLoss(mode="multiclass")

    def forward(self, preds, targets):
        ce_loss = self.ce(preds, targets)
        dice_loss = self.dice(preds, targets)
        return 0.5 * ce_loss + 0.5 * dice_loss