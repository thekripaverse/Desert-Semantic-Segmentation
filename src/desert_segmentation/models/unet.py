import torch.nn as nn
import segmentation_models_pytorch as smp


def build_unet(num_classes: int, encoder_weights: str = "imagenet") -> nn.Module:
    """
    Builds U-Net model with ResNet-18 encoder.
    """
    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=num_classes,
    )
    return model