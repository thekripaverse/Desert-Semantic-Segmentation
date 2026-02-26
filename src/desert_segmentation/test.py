import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from models.unet import build_unet
from datasets.desert_dataset import DesertDataset
from utils.metrics import compute_iou
from utils.tta import tta_predict


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = 10
    image_size = 256

    transform = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    val_dataset = DesertDataset(
        "val/Color_Images",
        "val/Segmentation",
        transform
    )

    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    model = build_unet(num_classes, encoder_weights=None).to(device)
    model.load_state_dict(torch.load("weights/best_model.pth", map_location=device))
    model.eval()

    total_iou = 0.0

    with torch.no_grad():
        for images, masks in tqdm(val_loader):
            images = images.to(device)
            masks = masks.to(device)

            outputs = tta_predict(model, images)
            total_iou += compute_iou(outputs, masks, num_classes)

    print("Final TTA mIoU:", total_iou / len(val_loader))


if __name__ == "__main__":
    main()