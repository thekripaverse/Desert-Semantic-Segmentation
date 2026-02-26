import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from typing import Optional

class SegmentationLoss(nn.Module):
    """
    Hybrid segmentation loss combining weighted Cross Entropy and Dice Loss.
    This helps address class imbalance in desert terrain data.
    """
    def __init__(self, class_weights: Optional[torch.Tensor] = None):
        """
        Args:
            class_weights: A manual rescaling weight given to each class.
        """
        super(SegmentationLoss, self).__init__()
        # CrossEntropy handles pixel-wise accuracy
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        # DiceLoss handles global boundary overlap
        self.dice = smp.losses.DiceLoss(mode="multiclass")

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Combines losses with a 50/50 ratio for stable convergence.
        """
        ce_loss = self.ce(preds, targets)
        dice_loss = self.dice(preds, targets)
        
        # Combining both losses ensures stable gradients and boundary awareness
        return 0.5 * ce_loss + 0.5 * dice_loss
