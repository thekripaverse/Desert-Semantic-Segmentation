import torch
from src.desert_segmentation.models.unet import build_unet


def test_model_output_shape():
    model = build_unet(num_classes=10, encoder_weights=None)
    model.eval()

    x = torch.randn(1, 3, 256, 256)
    output = model(x)

    assert output.shape == (1, 10, 256, 256)


def test_model_parameters():
    model = build_unet(num_classes=10, encoder_weights=None)
    params = sum(p.numel() for p in model.parameters())

    assert params > 0
