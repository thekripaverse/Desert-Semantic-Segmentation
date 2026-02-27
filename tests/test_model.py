"""
Unit tests for U-Net model construction.

These tests validate:
- Correct output tensor shape
- Proper parameter initialization
- Gradient flow capability
- Model structural integrity
"""

import torch
import pytest

from src.desert_segmentation.models.unet import build_unet


def test_model_output_shape():
    """
    The model should return logits of shape:
    (batch_size, num_classes, height, width)
    """

    model = build_unet(num_classes=10, encoder_weights=None)
    model.eval()

    x = torch.randn(1, 3, 256, 256)
    output = model(x)

    assert output.shape == (1, 10, 256, 256)


def test_model_has_parameters():
    """
    The model must contain trainable parameters.
    """

    model = build_unet(num_classes=10, encoder_weights=None)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    assert total_params > 0
    assert trainable_params > 0


def test_model_backward_pass():
    """
    The model should support gradient backpropagation.
    """

    model = build_unet(num_classes=10, encoder_weights=None)

    x = torch.randn(2, 3, 128, 128)
    target = torch.randint(0, 10, (2, 128, 128))

    output = model(x)

    loss = torch.nn.functional.cross_entropy(output, target)
    loss.backward()

    # Ensure at least one parameter received gradient
    grads = [p.grad for p in model.parameters() if p.grad is not None]

    assert len(grads) > 0


def test_model_forward_consistency():
    """
    Model should produce deterministic output shape
    regardless of input batch size.
    """

    model = build_unet(num_classes=10, encoder_weights=None)
    model.eval()

    for batch_size in [1, 2, 4]:
        x = torch.randn(batch_size, 3, 128, 128)
        output = model(x)
        assert output.shape[0] == batch_size
        assert output.shape[1] == 10
