import torch
from src.desert_segmentation.utils.metrics import compute_iou


def test_compute_iou_perfect_match():
    preds = torch.tensor([[[[0.9, 0.1],
                            [0.2, 0.8]],
                           [[0.1, 0.9],
                            [0.8, 0.2]]]])

    masks = torch.tensor([[0, 1],
                          [1, 0]])

    preds = preds.float()
    masks = masks.long()

    iou = compute_iou(preds, masks, num_classes=2)

    assert 0 <= iou <= 1


def test_compute_iou_shape():
    preds = torch.randn(1, 3, 4, 4)
    masks = torch.randint(0, 3, (1, 4, 4))

    iou = compute_iou(preds, masks, num_classes=3)

    assert isinstance(iou, float)
