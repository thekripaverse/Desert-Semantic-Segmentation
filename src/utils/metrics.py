import torch

def compute_iou(preds: torch.Tensor, masks: torch.Tensor, num_classes: int) -> float:
    preds = torch.argmax(preds, dim=1)
    ious = []

    for cls in range(num_classes):
        pred_cls = preds == cls
        mask_cls = masks == cls

        intersection = (pred_cls & mask_cls).sum().item()
        union = (pred_cls | mask_cls).sum().item()

        if union == 0:
            continue

        ious.append(intersection / union)

    return sum(ious) / len(ious)