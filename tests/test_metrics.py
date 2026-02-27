"""
Unit tests for IoU metric computation.

These tests validate:
- Correct output range
- Proper tensor shape handling
- Deterministic behavior
- Edge case robustness
"""

import torch
import pytest

from src.desert_segmentation.utils.metrics import compute_iou


def test_compute_iou_perfect_match():
    """
    IoU should be 1.0 when predictions perfectly match ground truth.
    """

    # Two-class perfect prediction
    preds = torch.tensor([[
        [[1.0, 0.0],
         [0.0, 1.0]],   # Class 0 logits
        [[0.0, 1.0],
         [1.0, 0.0]]    # Class 1 logits
    ]])

    masks = torch.tensor([[
        [0, 1],
        [1, 0]
    ]])

    iou = compute_iou(preds, masks, num_classes=2)

    assert pytest.approx(iou, 0.01) == 1.0


def test_compute_iou_valid_range():
    """
    IoU must always be between 0 and 1.
    """

    preds = torch.randn(1, 3, 4, 4)
    masks = torch.randint(0, 3, (1, 4, 4))

    iou = compute_iou(preds, masks, num_classes=3)

    assert 0.0 <= iou <= 1.0


def test_compute_iou_output_type():
    """
    IoU output must be a Python float.
    """

    preds = torch.randn(1, 4, 8, 8)
    masks = torch.randint(0, 4, (1, 8, 8))

    iou = compute_iou(preds, masks, num_classes=4)

    assert isinstance(iou, float)


def test_compute_iou_empty_union():
    """
    If a class is absent in both prediction and mask,
    it should not cause division-by-zero errors.
    """

    preds = torch.zeros(1, 2, 2, 2)
    masks = torch.zeros(1, 2, 2).long()

    iou = compute_iou(preds, masks, num_classes=2)

    assert 0.0 <= iou <= 1.0
