"""
Evaluation script for Desert Semantic Segmentation.

This script:
- Loads the trained best model checkpoint
- Applies validation preprocessing
- Performs inference with Test-Time Augmentation (TTA)
- Computes final mean Intersection over Union (mIoU)

Designed for clean reproducibility and inference benchmarking.
"""

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Project module imports
from models.unet import build_unet
from datasets.desert_dataset import DesertDataset
from utils.metrics import compute_iou
from utils.tta import tta_predict


def main():
    """
    Executes validation evaluation using the saved best model.
    Computes final TTA-based mIoU score.
    """

    # ===========================
    # Device Configuration
    # ===========================
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = 10
    image_size = 256

    # ===========================
    # Validation Transform Pipeline
    # ===========================
    transform = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    # ===========================
    # Dataset & DataLoader
    # ===========================
    val_dataset = DesertDataset(
        "val/Color_Images",
        "val/Segmentation",
        transform
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False
    )

    # ===========================
    # Model Initialization
    # ===========================
    model = build_unet(num_classes, encoder_weights=None).to(device)

    # Load trained weights
    model.load_state_dict(
        torch.load("weights/best_model.pth", map_location=device)
    )

    model.eval()

    # ===========================
    # Inference + Evaluation
    # ===========================
    total_iou = 0.0

    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="Evaluating"):

            images = images.to(device)
            masks = masks.to(device)

            # Apply Test-Time Augmentation inference
            outputs = tta_predict(model, images)

            # Compute batch IoU
            total_iou += compute_iou(outputs, masks, num_classes)

    final_iou = total_iou / len(val_loader)

    print(f"Final TTA mIoU: {final_iou:.4f}")


if __name__ == "__main__":
    main()
